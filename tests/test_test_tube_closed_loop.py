from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from demo2gripper.test_tube_closed_loop import (
    CLOSED_LOOP_CONFIG_SCHEMA,
    CLOSED_LOOP_REPLAY_SCHEMA,
    ActionKind,
    ClosedLoopContractError,
    ClosedLoopObservation,
    ClosedLoopPolicy,
    GripperFeedback,
    NormalizedDemoGraspRelation,
    ObjectPoseEstimate,
    PoseBackendDescriptor,
    PoseCapability,
    RobotFeedback,
    TactileFeedback,
    bounded_visual_servo,
    canonicalize_cylindrical_tube_pose,
    grounded_grasp_target,
    held_tube_level_target,
    normalized_top_down_grasp_target,
    top_pregrasp_target,
    tube_horizontal_error_deg,
)
from demo2gripper.test_tube_closed_loop import (
    TestTubeClosedLoopController as TubeClosedLoopController,
)
from demo2gripper.test_tube_closed_loop import (
    TestTubeClosedLoopGeometry as TubeClosedLoopGeometry,
)
from demo2gripper.test_tube_preflight import TubeState


def _transform(
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_deg_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", rotation_deg_xyz, degrees=True).as_matrix()
    result[:3, 3] = translation
    return result


def _backend(*, qualified: bool = True, fallback: bool = False) -> PoseBackendDescriptor:
    capabilities = (
        {
            PoseCapability.METRIC_TRANSLATION,
            PoseCapability.IMAGE_AXIS,
            PoseCapability.SOURCE_TUBE,
        }
        if fallback
        else set(PoseCapability)
    )
    return PoseBackendDescriptor(
        name="rgbd_axis_anchor" if fallback else "replay_6d",
        available=True,
        runtime_qualified=qualified,
        qualification_id="qualification-001" if qualified else None,
        capabilities=frozenset(capabilities),
    )


def _pose(
    object_id: str,
    now_s: float,
    transform: np.ndarray | None = None,
    *,
    backend: PoseBackendDescriptor | None = None,
    occluded: bool = False,
) -> ObjectPoseEstimate:
    tube = object_id == "tube"
    return ObjectPoseEstimate(
        object_id=object_id,
        T_base_object=np.eye(4) if transform is None else transform,
        observed_at_s=now_s - 0.01,
        position_std_m=0.001,
        orientation_std_deg=1.0,
        backend=backend or _backend(),
        tracking_id=f"{object_id}-track",
        inference_latency_s=0.01,
        mask_id=f"{object_id}-mask",
        mask_iou=0.2 if occluded else 0.9,
        valid_depth_count=0 if occluded else 200,
        foreground_depth_support_count=0 if occluded else 120,
        foreground_depth_support_fraction=0.0 if occluded else 0.6,
        background_depth_fraction=1.0 if occluded else 0.1,
        masked_depth_cluster_qualified=not occluded,
        axis_sign_source="open_end_detector" if tube else "not_applicable",
        cylindrical_roll_canonicalized=tube,
        occluded_by_gripper=occluded,
    )


def _geometry() -> TubeClosedLoopGeometry:
    return TubeClosedLoopGeometry(
        relation_id="demo-relation-001",
        T_tube_tcp_demo_grasp=np.eye(4),
        T_socket_tube_entry=np.eye(4),
    )


def _observation(
    now_s: float,
    *,
    tcp: np.ndarray | None = None,
    tube_pose: ObjectPoseEstimate | None = None,
    socket_pose: ObjectPoseEstimate | None = None,
    moving: bool = False,
    motion_complete: bool = True,
    bounded_servo_supported: bool = True,
    width_m: float = 0.04,
    open_confirmed: bool = False,
    tactile: TactileFeedback | None = None,
    **flags,
) -> ClosedLoopObservation:
    return ClosedLoopObservation(
        observation_id=f"obs-{now_s:.3f}",
        now_s=now_s,
        robot=RobotFeedback(
            T_base_tcp=np.eye(4) if tcp is None else tcp,
            motion_complete=motion_complete,
            bounded_tcp_servo_supported=bounded_servo_supported,
        ),
        gripper=GripperFeedback(
            width_m=width_m,
            moving=moving,
            fault_free=True,
            open_confirmed=open_confirmed,
        ),
        tactile=tactile
        or TactileFeedback(valid=True, observed_at_s=now_s - 0.01),
        tube_pose=tube_pose,
        socket_pose=socket_pose,
        **flags,
    )


def _final_approach_controller() -> TubeClosedLoopController:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(final_approach_commit_distance_m=0.02),
    )
    controller.state = TubeState.SIDE_GRASP
    first = controller.step(_observation(
        1.0, tcp=_transform((0, 0, 0.018)), tube_pose=_pose("tube", 1.0), width_m=0.0494,
    ))
    assert first.action.kind is ActionKind.SERVO_TCP
    assert "final_contact_approach" not in first.action.parameters
    second = controller.step(_observation(
        1.1, tcp=_transform((0, 0, 0.016)), tube_pose=_pose("tube", 1.1), width_m=0.0494,
    ))
    assert second.action.parameters["final_contact_approach"] is True
    assert np.allclose(second.action.parameters["final_contact_target"], np.eye(4))
    assert second.action.parameters["committed_source_observed_at_s"] == pytest.approx(1.09)
    command = second.action.parameters["T_base_tcp_command"]
    assert np.linalg.norm(command[:3, 3] - [0, 0, 0.016]) <= 0.008 + 1e-12
    return controller


def _startup_recovery_controller() -> TubeClosedLoopController:
    return TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(observe_timeout_s=2.0, approach_timeout_s=60.0),
        startup_recovery_path=(
            _transform((0.0, 0.0, 0.18)),
            _transform((0.0, 0.0, 0.196)),
            _transform((0.0, 0.0, 0.23)),
        ),
    )


def test_startup_recovery_reverses_known_path_then_reacquires_without_object_pose() -> None:
    controller = _startup_recovery_controller()
    waiting = controller.step(_observation(
        1.0, tcp=_transform((0, 0, 0.18)), width_m=0.0494,
        tactile=TactileFeedback(valid=False, observed_at_s=0.99),
    ))
    assert waiting.action.kind is ActionKind.COLLECT_EVIDENCE
    current = _transform((0, 0, 0.18))
    seen_indices = set()
    for step in range(15):
        decision = controller.step(_observation(
            1.1 + step * 0.1, tcp=current, width_m=0.0494,
        ))
        if decision.action.kind is ActionKind.PAUSE_ARM:
            assert decision.action.parameters["startup_recovery_complete"] is True
            break
        assert decision.action.kind is ActionKind.SERVO_TCP
        assert decision.action.parameters["startup_recovery"] is True
        seen_indices.add(decision.action.parameters["recovery_path_index"])
        command = decision.action.parameters["T_base_tcp_command"]
        assert np.linalg.norm(command[:3, 3] - current[:3, 3]) <= 0.008 + 1e-12
        assert command[2, 3] >= current[2, 3]
        current = command
    else:
        pytest.fail("recorded reverse approach never completed")
    assert seen_indices == {1, 2}
    assert current[2, 3] == pytest.approx(0.23, abs=0.0025)
    assert controller.state is TubeState.OBSERVE
    missing = controller.step(_observation(3.0, tcp=current, width_m=0.0494))
    assert missing.action.kind is ActionKind.COLLECT_EVIDENCE
    qualified = controller.step(_observation(
        3.1, tcp=current, width_m=0.0494,
        tube_pose=_pose("tube", 3.1), socket_pose=_pose("socket", 3.1),
    ))
    assert qualified.next_state is TubeState.APPROACH_TUBE
    assert qualified.action.kind is ActionKind.MOVE_TCP


def test_startup_recovery_waits_for_live_stationary_feedback_before_reverse() -> None:
    controller = _startup_recovery_controller()
    waiting = controller.step(_observation(
        1.0,
        tcp=_transform((0, 0, 0.18)),
        width_m=0.0494,
        motion_complete=False,
    ))
    assert waiting.action.kind is ActionKind.COLLECT_EVIDENCE
    assert controller._startup_recovery_armed is False

    started = controller.step(_observation(
        1.1,
        tcp=_transform((0, 0, 0.18)),
        width_m=0.0494,
        motion_complete=True,
    ))
    assert started.action.kind is ActionKind.SERVO_TCP
    assert started.action.parameters["startup_recovery"] is True


@pytest.mark.parametrize("case", [
    "wrong_position", "wrong_rotation", "contact", "single_contact", "slip",
    "closed", "moving", "gripper_object", "robot_fault", "no_servo",
])
def test_startup_recovery_never_starts_from_unverified_or_loaded_state(case: str) -> None:
    controller = _startup_recovery_controller()
    obs = _observation(1.0, tcp=_transform((0, 0, 0.18)), width_m=0.0494)
    if case == "wrong_position":
        obs = replace(obs, robot=replace(obs.robot, T_base_tcp=_transform((0.003, 0, 0.18))))
    elif case == "wrong_rotation":
        obs = replace(obs, robot=replace(
            obs.robot, T_base_tcp=_transform((0, 0, 0.18), (0, 0, 2.1)),
        ))
    elif case == "contact":
        obs = replace(obs, tactile=replace(obs.tactile, contact_present=True))
    elif case == "single_contact":
        obs = replace(obs, tactile=replace(obs.tactile, single_contact_side="left"))
    elif case == "slip":
        obs = replace(obs, tactile=replace(obs.tactile, slip_detected=True))
    elif case == "closed":
        obs = replace(obs, gripper=replace(obs.gripper, width_m=0.0479))
    elif case == "moving":
        obs = replace(obs, gripper=replace(obs.gripper, moving=True))
    elif case == "gripper_object":
        obs = replace(obs, gripper=replace(obs.gripper, object_detected=True))
    elif case == "robot_fault":
        obs = replace(obs, robot=replace(obs.robot, fault="reflex"))
    else:
        obs = replace(obs, robot=replace(obs.robot, bounded_tcp_servo_supported=False))
    decision = controller.step(obs)
    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD


@pytest.mark.parametrize("case", ["lost_tactile", "contact", "path_deviation"])
def test_startup_recovery_holds_on_new_contact_feedback_loss_or_path_departure(case: str) -> None:
    controller = _startup_recovery_controller()
    started = controller.step(_observation(
        1.0, tcp=_transform((0, 0, 0.18)), width_m=0.0494,
    ))
    assert started.action.kind is ActionKind.SERVO_TCP
    obs = _observation(1.1, tcp=_transform((0, 0, 0.188)), width_m=0.0494)
    if case == "lost_tactile":
        obs = replace(obs, tactile=replace(obs.tactile, valid=False))
    elif case == "contact":
        obs = replace(obs, tactile=replace(obs.tactile, contact_present=True))
    else:
        obs = replace(obs, robot=replace(obs.robot, T_base_tcp=_transform((0.009, 0, 0.188))))
    assert controller.step(obs).next_state is TubeState.HOLD


def test_startup_recovery_deadline_is_fixed_across_waypoints() -> None:
    controller = _startup_recovery_controller()
    assert controller.step(_observation(
        1.0, tcp=_transform((0, 0, 0.18)), width_m=0.0494,
    )).action.kind is ActionKind.SERVO_TCP
    progressed = controller.step(_observation(
        31.0, tcp=_transform((0, 0, 0.196)), width_m=0.0494,
    ))
    assert progressed.action.parameters["recovery_path_index"] == 2
    assert controller._state_started_s == 1.0
    expired = controller.step(_observation(
        61.1, tcp=_transform((0, 0, 0.204)), width_m=0.0494,
    ))
    assert expired.next_state is TubeState.HOLD
    assert "timeout" in expired.action.reason


def test_startup_recovery_flows_through_intermediate_pose_but_stops_at_final_pose() -> None:
    controller = _startup_recovery_controller()
    controller.step(_observation(1.0, tcp=_transform((0, 0, 0.18)), width_m=0.0494))
    progressing = controller.step(_observation(
        1.1, tcp=_transform((0, 0, 0.194)), width_m=0.0494, motion_complete=False,
    ))
    assert progressing.action.kind is ActionKind.SERVO_TCP
    assert progressing.action.parameters["recovery_path_index"] == 2
    assert progressing.action.parameters["T_base_tcp_command"][2, 3] > 0.196
    assert controller._state_started_s == 1.0
    arriving = controller.step(_observation(
        1.2, tcp=_transform((0, 0, 0.229)), width_m=0.0494, motion_complete=False,
    ))
    assert arriving.action.kind is ActionKind.SERVO_TCP
    assert not controller._startup_recovery_complete
    settled = controller.step(_observation(
        1.3, tcp=_transform((0, 0, 0.23)), width_m=0.0494, motion_complete=True,
    ))
    assert settled.action.kind is ActionKind.PAUSE_ARM
    assert settled.action.parameters["startup_recovery_complete"] is True


def test_startup_recovery_path_requires_two_rigid_poses_and_copies_input() -> None:
    with pytest.raises(ClosedLoopContractError):
        TubeClosedLoopController(geometry=_geometry(), startup_recovery_path=[np.eye(4)])
    malformed = np.eye(4)
    malformed[0, 0] = 2.0
    with pytest.raises(ValueError):
        TubeClosedLoopController(
            geometry=_geometry(), startup_recovery_path=[np.eye(4), malformed],
        )
    expected, endpoint = _transform((0, 0, 0.18)), _transform((0, 0, 0.196))
    controller = TubeClosedLoopController(
        geometry=_geometry(), startup_recovery_path=[expected, endpoint],
    )
    endpoint[2, 3] = 0.0
    decision = controller.step(_observation(1.0, tcp=expected, width_m=0.0494))
    assert decision.action.parameters["T_base_tcp_live_target"][2, 3] == pytest.approx(0.196)


def test_committed_final_approach_keeps_endpoint_and_timestamp_until_measured_close() -> None:
    controller = _final_approach_controller()
    # The original camera capture is older than pose_max_age_s. This is the
    # already committed finite motion, with fresh RGB identity and tactile.
    advance = controller.step(_observation(
        2.5, tcp=_transform((0, 0, 0.007)), width_m=0.0494,
        source_depth_only_loss=True, motion_complete=False,
    ))
    assert advance.action.kind is ActionKind.SERVO_TCP
    assert advance.action.parameters["committed_source_observed_at_s"] == pytest.approx(1.09)
    assert np.allclose(advance.action.parameters["final_contact_target"], np.eye(4))
    # Being inside the old 8 mm occlusion window alone does not close.
    assert controller._phase == "final_contact_approach"
    first_settle = controller.step(_observation(
        2.6, tcp=_transform((0, 0, 0.001)), width_m=0.0494, source_depth_only_loss=True,
    ))
    assert first_settle.action.kind is ActionKind.PAUSE_ARM
    assert first_settle.action.parameters["final_contact_arrived"] is True
    assert first_settle.action.parameters["zero_active_servo"] is True
    assert first_settle.action.parameters["committed_source_observed_at_s"] == pytest.approx(1.09)
    # The finger now hides RGB too. The arm has already arrived and stopped;
    # only stationary FK and fresh no-contact tactile are needed before close.
    closed = controller.step(_observation(
        2.7, tcp=_transform((0, 0, 0.001)), width_m=0.0494, source_depth_only_loss=False,
    ))
    assert closed.action.kind is ActionKind.START_CONTINUOUS_CLOSE
    assert closed.action.parameters["committed_final_approach"] is True
    assert closed.action.parameters["zero_active_servo"] is True
    assert closed.action.parameters["committed_source_observed_at_s"] == pytest.approx(1.09)
    assert np.allclose(closed.action.parameters["final_contact_target"], np.eye(4))
    assert controller._state_started_s == 2.7


def test_committed_final_approach_accepts_current_rgb_identity_when_axis_sign_drops() -> None:
    controller = _final_approach_controller()
    continued = controller.step(_observation(
        2.5,
        tcp=_transform((0, 0, 0.003)),
        width_m=0.0494,
        source_depth_only_loss=False,
        source_rgb_identity_retained=True,
        motion_complete=False,
    ))
    assert continued.action.kind is ActionKind.SERVO_TCP
    assert continued.action.parameters["final_contact_approach"] is True
    assert continued.action.parameters["committed_source_observed_at_s"] == pytest.approx(1.09)


@pytest.mark.parametrize("case", [
    "arrival_drift", "goal_drift", "rotation_drift", "still_moving", "invalid_tactile",
    "contact", "slip", "gripper_moved", "identity_changed", "target_moved", "timeout",
])
def test_arrived_final_approach_never_closes_after_stationary_evidence_breaks(case: str) -> None:
    controller = _final_approach_controller()
    arrival = _transform((0, 0, 0.002), (0, 0, 1.5))
    pause = controller.step(_observation(
        1.2, tcp=arrival, width_m=0.0494, source_depth_only_loss=True,
    ))
    assert pause.action.parameters["final_contact_arrived"] is True
    now = 9.11 if case == "timeout" else 1.3
    obs = _observation(now, tcp=arrival, width_m=0.0494)
    if case == "arrival_drift":
        obs = replace(obs, robot=replace(
            obs.robot, T_base_tcp=_transform((0, 0, -0.002), (0, 0, 1.5)),
        ))
    elif case == "goal_drift":
        obs = replace(obs, robot=replace(obs.robot, T_base_tcp=_transform((0, 0, 0.003))))
    elif case == "rotation_drift":
        obs = replace(obs, robot=replace(
            obs.robot, T_base_tcp=_transform((0, 0, 0.002), (0, 0, -1.5)),
        ))
    elif case == "still_moving":
        obs = replace(obs, robot=replace(obs.robot, motion_complete=False))
    elif case == "invalid_tactile":
        obs = replace(obs, tactile=replace(obs.tactile, valid=False))
    elif case == "contact":
        obs = replace(obs, tactile=replace(obs.tactile, contact_present=True))
    elif case == "slip":
        obs = replace(obs, tactile=replace(obs.tactile, slip_detected=True))
    elif case == "gripper_moved":
        obs = replace(obs, gripper=replace(obs.gripper, moving=True))
    elif case == "identity_changed":
        obs = replace(obs, tube_pose=replace(_pose("tube", now), tracking_id="new-tube"))
    elif case == "target_moved":
        obs = replace(obs, tube_pose=_pose("tube", now, _transform((0.004, 0, 0))))
    held = controller.step(obs)
    assert held.action.kind is ActionKind.STOP_AND_HOLD
    assert held.next_state is TubeState.HOLD


def test_final_arrival_repeated_settle_pauses_without_new_servo_or_timer_reset() -> None:
    controller = _final_approach_controller()
    controller.policy = replace(controller.policy, servo_settle_windows=3)
    first = controller.step(_observation(
        1.2, tcp=_transform((0, 0, 0.001)), width_m=0.0494,
        tube_pose=_pose("tube", 1.2),
    ))
    assert first.action.kind is ActionKind.PAUSE_ARM
    second = controller.step(_observation(
        1.3, tcp=_transform((0, 0, 0.001)), width_m=0.0494,
    ))
    assert second.action.kind is ActionKind.PAUSE_ARM
    assert controller._state_started_s == 1.1
    assert second.action.parameters["committed_source_observed_at_s"] == pytest.approx(1.09)
    third = controller.step(_observation(
        1.4, tcp=_transform((0, 0, 0.001)), width_m=0.0494,
    ))
    assert third.action.kind is ActionKind.START_CONTINUOUS_CLOSE


def test_committed_final_approach_missing_rgb_pauses_without_extending_deadline() -> None:
    controller = _final_approach_controller()
    pause = controller.step(_observation(2.0, width_m=0.0494))
    assert pause.action.kind is ActionKind.PAUSE_ARM
    assert pause.action.parameters["zero_active_servo"] is True
    assert controller._state_started_s == 1.1
    expired = controller.step(_observation(
        9.11, width_m=0.0494, source_depth_only_loss=True,
    ))
    assert expired.action.kind is ActionKind.STOP_AND_HOLD
    assert "timeout" in expired.action.reason


@pytest.mark.parametrize("case", [
    "moved", "rotated", "identity", "uncertainty", "stale_pose", "contact",
    "invalid_tactile", "gripper_moved", "robot_fault", "outside_committed_path",
])
def test_committed_final_approach_rejects_new_unsafe_evidence(case: str) -> None:
    controller = _final_approach_controller()
    now = 1.2
    pose = _pose("tube", now)
    kwargs = {"tube_pose": pose, "width_m": 0.0494, "tcp": _transform((0, 0, 0.01))}
    if case == "moved":
        kwargs["tube_pose"] = replace(pose, T_base_object=_transform((0.004, 0, 0)))
    elif case == "rotated":
        kwargs["tube_pose"] = replace(pose, T_base_object=_transform(rotation_deg_xyz=(0, 0, 3)))
    elif case == "identity":
        kwargs["tube_pose"] = replace(pose, tracking_id="another-tube")
    elif case == "uncertainty":
        kwargs["tube_pose"] = replace(pose, position_std_m=0.02)
    elif case == "stale_pose":
        kwargs["tube_pose"] = replace(pose, observed_at_s=0.5)
    elif case == "contact":
        kwargs["tactile"] = TactileFeedback(valid=True, observed_at_s=now, contact_present=True)
    elif case == "invalid_tactile":
        kwargs["tactile"] = TactileFeedback(valid=False, observed_at_s=now)
    elif case == "gripper_moved":
        kwargs["moving"] = True
    elif case == "outside_committed_path":
        kwargs["tcp"] = _transform((0.03, 0, 0.01))
    obs = _observation(now, **kwargs)
    if case == "robot_fault":
        obs = replace(obs, robot=replace(obs.robot, fault="protective_stop"))
    held = controller.step(obs)
    assert held.action.kind is ActionKind.STOP_AND_HOLD
    assert held.next_state is TubeState.HOLD


@pytest.mark.parametrize("case", [
    "unstable", "one_frame", "stale", "orientation", "not_open", "moving", "tactile",
])
def test_final_approach_commit_requires_stable_qualified_targets_and_open_hand(case: str) -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(), policy=ClosedLoopPolicy(final_approach_commit_distance_m=0.02),
    )
    controller.state = TubeState.SIDE_GRASP
    if case != "one_frame":
        controller.step(_observation(
            1.0, tcp=_transform((0, 0, 0.018)), tube_pose=_pose("tube", 1.0), width_m=0.0494,
        ))
    kwargs = {"tcp": _transform((0, 0, 0.016)), "tube_pose": _pose("tube", 1.1),
              "width_m": 0.0494}
    if case == "unstable":
        kwargs["tube_pose"] = _pose("tube", 1.1, _transform((0.004, 0, 0)))
    elif case == "stale":
        kwargs["tube_pose"] = _pose("tube", 0.5)
    elif case == "orientation":
        kwargs["tcp"] = _transform((0, 0, 0.016), (0, 0, 3))
    elif case == "not_open":
        kwargs["width_m"] = 0.04
    elif case == "moving":
        kwargs["moving"] = True
    elif case == "tactile":
        kwargs["tactile"] = TactileFeedback(valid=False, observed_at_s=1.1)
    result = controller.step(_observation(1.1, **kwargs))
    assert result.action.parameters.get("final_contact_approach") is not True
    assert controller._phase != "final_contact_approach"


def test_final_approach_commits_fresh_reacquisition_after_long_visual_gap() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(
            final_approach_commit_distance_m=0.02, approach_timeout_s=60.0,
        ),
    )
    controller.state = TubeState.SIDE_GRASP
    controller.step(_observation(
        1.0, tcp=_transform((0, 0, 0.018)), tube_pose=_pose("tube", 1.0), width_m=0.0494,
    ))
    paused = controller.step(_observation(
        20.0, tcp=_transform((0, 0, 0.012)), width_m=0.0494,
    ))
    assert paused.action.kind is ActionKind.PAUSE_ARM
    assert controller._last_qualified_grasp_observed_at_s == pytest.approx(0.99)
    reacquired = controller.step(_observation(
        24.58, tcp=_transform((0, 0, 0.012)), tube_pose=_pose("tube", 24.58), width_m=0.0494,
    ))
    assert reacquired.action.kind is ActionKind.SERVO_TCP
    assert reacquired.action.parameters["final_contact_approach"] is True
    assert reacquired.action.parameters["committed_source_observed_at_s"] == pytest.approx(24.57)


def test_side_grasp_descent_uses_approach_budget_before_close_begins() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(approach_timeout_s=60, close_timeout_s=1),
    )
    controller.state = TubeState.SIDE_GRASP
    controller.step(_observation(1, tcp=_transform((0, 0, 0.04)), tube_pose=_pose("tube", 1)))
    continued = controller.step(_observation(
        4, tcp=_transform((0, 0, 0.03)), tube_pose=_pose("tube", 4),
    ))
    assert continued.action.kind is ActionKind.SERVO_TCP
    assert continued.next_state is TubeState.SIDE_GRASP


def test_pregrasp_transition_commands_first_descent_not_another_standoff() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(), policy=ClosedLoopPolicy(servo_settle_windows=1),
    )
    controller.state = TubeState.APPROACH_TUBE
    result = controller.step(_observation(
        1, tcp=_transform((0, 0, 0.06)), tube_pose=_pose("tube", 1),
    ))
    assert result.next_state is TubeState.SIDE_GRASP
    assert result.action.parameters["T_base_tcp_command"][2, 3] == pytest.approx(0.052)
    assert result.action.parameters["translation_error_m"] == pytest.approx(0.06)


def test_reached_standoff_hands_depth_loss_to_bounded_cached_final_approach() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(
            final_approach_commit_distance_m=0.065,
            servo_settle_windows=2,
            pose_max_age_s=0.2,
        ),
    )
    controller.state = TubeState.APPROACH_TUBE
    visible = controller.step(_observation(
        1.0,
        tcp=_transform((0, 0, 0.08)),
        tube_pose=_pose("tube", 1.0),
        width_m=0.0494,
    ))
    assert visible.action.kind is ActionKind.SERVO_TCP

    handed_off = controller.step(_observation(
        2.0,
        tcp=_transform((0, 0, 0.06)),
        width_m=0.0494,
        source_depth_only_loss=True,
    ))

    assert handed_off.next_state is TubeState.SIDE_GRASP
    assert handed_off.action.kind is ActionKind.SERVO_TCP
    assert handed_off.action.parameters["final_contact_approach"] is True
    assert handed_off.action.parameters["depth_loss_handoff"] is True
    assert handed_off.action.parameters["committed_source_observed_at_s"] == pytest.approx(0.99)
    assert "current_rgb_identity_retained" in handed_off.predicates


@pytest.mark.parametrize("failure", ["not_rgb", "off_standoff", "contact", "not_open"])
def test_depth_loss_handoff_rejects_unqualified_standoff(failure: str) -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(final_approach_commit_distance_m=0.065),
    )
    controller.state = TubeState.APPROACH_TUBE
    controller.step(_observation(
        1.0,
        tcp=_transform((0, 0, 0.08)),
        tube_pose=_pose("tube", 1.0),
        width_m=0.0494,
    ))
    kwargs = {
        "tcp": _transform((0, 0, 0.06)),
        "width_m": 0.0494,
        "source_depth_only_loss": True,
    }
    if failure == "not_rgb":
        kwargs["source_depth_only_loss"] = False
    elif failure == "off_standoff":
        kwargs["tcp"] = _transform((0.01, 0, 0.06))
    elif failure == "contact":
        kwargs["tactile"] = TactileFeedback(
            valid=True, observed_at_s=1.99, contact_present=True
        )
    elif failure == "not_open":
        kwargs["width_m"] = 0.04
    decision = controller.step(_observation(2.0, **kwargs))
    assert decision.action.kind is ActionKind.PAUSE_ARM
    assert decision.next_state is TubeState.APPROACH_TUBE


def test_socket_standoff_transition_emits_bounded_adapter_servo_command() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(), policy=ClosedLoopPolicy(servo_settle_windows=1),
    )
    controller.state = TubeState.APPROACH_SOCKET
    controller._T_tube_tcp_held = np.eye(4)
    result = controller.step(_observation(
        1, tcp=_transform((0, 0, 0.05)), tube_pose=_pose("tube", 1),
        socket_pose=_pose("socket", 1),
    ))
    assert result.next_state is TubeState.ALIGN_SOCKET
    assert result.action.kind is ActionKind.SERVO_TCP
    # LiveIO consumes this exact command key, not the coarse MoveIt target key.
    command = result.action.parameters["T_base_tcp_command"]
    assert command[2, 3] == pytest.approx(0.042)
    assert np.allclose(result.action.parameters["T_base_tcp_live_target"], np.eye(4))
    assert result.action.parameters["bounded"] is True
    assert result.action.parameters["translation_error_m"] == pytest.approx(0.05)


def test_top_pregrasp_preserves_full_demo_relative_orientation() -> None:
    relation = _transform((0.01, -0.02, 0.03), (20.0, -15.0, 35.0))
    tube = _pose("tube", 1.0, _transform((0.5, 0.1, 0.2), (5.0, 10.0, 25.0)))
    geometry = TubeClosedLoopGeometry(
        relation_id="full-demo-relation",
        T_tube_tcp_demo_grasp=relation,
        T_socket_tube_entry=np.eye(4),
    )

    grasp = grounded_grasp_target(tube, geometry)
    pregrasp = top_pregrasp_target(grasp, 0.06)

    assert np.allclose(grasp, tube.T_base_object @ relation)
    assert np.allclose(pregrasp[:3, :3], grasp[:3, :3])
    assert np.allclose(pregrasp[:3, 3] - grasp[:3, 3], [0.0, 0.0, 0.06])


def test_normalized_relation_stays_top_down_for_a_rotated_live_tube() -> None:
    relation = NormalizedDemoGraspRelation(
        axial_fraction_closed_to_open=0.664323374340949,
        tube_length_m=0.110,
        tube_length_source="caliper-receipt-001",
        preferred_jaw_axis_base=np.array([-0.8893638, 0.32929212, 0.31717304]),
    )
    first_tube = _transform((0.4, 0.1, 0.2), (0.0, 60.0, 10.0))
    second_tube = _transform((0.5, -0.1, 0.3), (70.0, 0.0, 80.0))

    first = normalized_top_down_grasp_target(first_tube, relation)
    second = normalized_top_down_grasp_target(second_tube, relation)

    np.testing.assert_allclose(first[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(second[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)
    assert abs(float(first[:3, 0] @ first_tube[:3, 2])) < 1e-12
    assert abs(float(second[:3, 0] @ second_tube[:3, 2])) < 1e-12
    expected_offset = (0.664323374340949 - 0.5) * 0.110
    np.testing.assert_allclose(
        first[:3, 3],
        first_tube[:3, 3] + expected_offset * first_tube[:3, 2],
    )
    np.testing.assert_allclose(
        second[:3, 3],
        second_tube[:3, 3] + expected_offset * second_tube[:3, 2],
    )


def test_geometry_accepts_runtime_normalized_relation_without_demo_6d_pose() -> None:
    geometry = TubeClosedLoopGeometry(
        relation_id="normalized-demo",
        T_tube_tcp_demo_grasp=None,
        T_socket_tube_entry=np.eye(4),
        normalized_demo_grasp=NormalizedDemoGraspRelation(
            axial_fraction_closed_to_open=0.664323374340949,
            tube_length_m=0.110,
            tube_length_source="caliper-receipt-001",
            preferred_jaw_axis_base=np.array([1.0, 0.0, 0.0]),
        ),
    )
    target = grounded_grasp_target(
        _pose("tube", 1.0, _transform((0.4, 0.1, 0.2), (0.0, 45.0, 0.0))),
        geometry,
    )
    np.testing.assert_allclose(target[:3, 2], [0.0, 0.0, -1.0], atol=1e-12)


def test_cylinder_roll_is_canonical_but_open_end_sign_is_not_discarded() -> None:
    first = _transform((0.4, 0.2, 0.1), (0.0, 0.0, 10.0))
    second = _transform((0.4, 0.2, 0.1), (0.0, 0.0, 120.0))

    canonical_first = canonicalize_cylindrical_tube_pose(
        first, long_axis_index=2, open_end_sign=1
    )
    canonical_second = canonicalize_cylindrical_tube_pose(
        second, long_axis_index=2, open_end_sign=1
    )
    flipped = canonicalize_cylindrical_tube_pose(
        first, long_axis_index=2, open_end_sign=-1
    )

    assert np.allclose(canonical_first, canonical_second)
    assert np.allclose(flipped[:3, 2], -canonical_first[:3, 2])
    assert np.linalg.det(flipped[:3, :3]) == pytest.approx(1.0)


def test_visual_servo_caps_each_step_and_rejects_target_jump() -> None:
    policy = ClosedLoopPolicy(
        servo_max_translation_step_m=0.008,
        servo_max_rotation_step_deg=4.0,
        servo_max_target_jump_m=0.015,
    )
    measured = np.eye(4)
    target = _transform((0.05, 0.0, 0.0), (0.0, 0.0, 20.0))
    result = bounded_visual_servo(
        T_base_tcp_measured=measured,
        T_base_tcp_target=target,
        T_base_tcp_initial_target=target,
        previous_target=target,
        policy=policy,
    )
    assert np.linalg.norm(result.command_target[:3, 3]) == pytest.approx(0.008)
    command_rotation_deg = math.degrees(
        Rotation.from_matrix(result.command_target[:3, :3]).magnitude()
    )
    assert command_rotation_deg == pytest.approx(4.0)

    jumped = target.copy()
    jumped[1, 3] += 0.020
    rejected = bounded_visual_servo(
        T_base_tcp_measured=measured,
        T_base_tcp_target=jumped,
        T_base_tcp_initial_target=target,
        previous_target=target,
        policy=policy,
    )
    assert "live_target_translation_jump_exceeded" in rejected.blockers


def test_held_tube_level_target_uses_live_axis_and_measured_tcp() -> None:
    tube = _transform((0.45, 0.02, 0.18), (35.0, 0.0, 20.0))
    tcp = _transform((0.46, 0.03, 0.22), (10.0, -15.0, 40.0))
    relation = np.linalg.inv(tube) @ tcp

    target_tcp = held_tube_level_target(
        T_base_tube=tube,
        T_base_tcp_measured=tcp,
        min_horizontal_projection=0.10,
    )
    predicted_tube = target_tcp @ np.linalg.inv(relation)

    np.testing.assert_allclose(target_tcp[:3, 3], tcp[:3, 3])
    assert tube_horizontal_error_deg(predicted_tube) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(
        np.linalg.inv(predicted_tube) @ target_tcp,
        relation,
        atol=1e-10,
    )


def test_held_tube_level_target_fails_closed_without_horizontal_azimuth() -> None:
    with pytest.raises(
        ClosedLoopContractError,
        match="held_tube_horizontal_direction_unobservable",
    ):
        held_tube_level_target(
            T_base_tube=np.eye(4),
            T_base_tcp_measured=np.eye(4),
            min_horizontal_projection=0.10,
        )


def test_level_state_debounces_geometry_and_robot_convergence_before_opening() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(level_settle_windows=2),
    )
    controller.state = TubeState.LEVEL_HELD_TUBE
    tilted = _transform(rotation_deg_xyz=(30.0, 0.0, 0.0))
    contact = TactileFeedback(
        valid=True,
        observed_at_s=0.99,
        contact_present=True,
        bilateral_contact=True,
    )

    first = controller.step(
        _observation(1.0, tcp=tilted, tube_pose=_pose("tube", 1.0, tilted), tactile=contact)
    )
    assert first.next_state is TubeState.LEVEL_HELD_TUBE
    assert first.action.kind is ActionKind.SERVO_TCP
    target = np.asarray(first.action.parameters["T_base_tcp_live_target"])

    first_stable = controller.step(
        _observation(
            1.1,
            tcp=target,
            tube_pose=_pose("tube", 1.1, target),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.09,
                contact_present=True,
                bilateral_contact=True,
            ),
        )
    )
    assert first_stable.next_state is TubeState.LEVEL_HELD_TUBE
    assert first_stable.action.kind is ActionKind.SERVO_TCP
    assert first_stable.action.parameters["level_confirmation_windows"] == 1

    confirmed = controller.step(
        _observation(
            1.2,
            tcp=target,
            tube_pose=_pose("tube", 1.2, target),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.19,
                contact_present=True,
                bilateral_contact=True,
            ),
        )
    )
    assert confirmed.next_state is TubeState.PIVOT_MICRO_LOOSEN
    assert confirmed.action.kind is ActionKind.OPEN_GRIPPER_STEP
    assert confirmed.action.parameters["level_confirmation_windows"] == 2


def test_level_state_uses_one_lifted_moveit_escape_after_progress_stalls() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(
            level_stall_confirmation_windows=2,
            level_stall_min_progress_deg=0.5,
            level_escape_lift_m=0.050,
        ),
    )
    controller.state = TubeState.LEVEL_HELD_TUBE
    tilted = _transform((0.45, 0.02, 0.24), (30.0, 0.0, 0.0))

    def stalled_observation(now_s: float) -> ClosedLoopObservation:
        return _observation(
            now_s,
            tcp=tilted,
            tube_pose=_pose("tube", now_s, tilted),
            motion_complete=False,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=now_s - 0.01,
                contact_present=True,
                bilateral_contact=True,
                bilateral_contact_interior=True,
            ),
        )

    first = controller.step(stalled_observation(1.0))
    second = controller.step(stalled_observation(1.1))
    escaped = controller.step(stalled_observation(1.2))

    assert first.action.kind is ActionKind.SERVO_TCP
    assert second.action.kind is ActionKind.SERVO_TCP
    assert escaped.action.kind is ActionKind.MOVE_TCP
    assert escaped.action.parameters["plan_with_moveit"] is True
    assert escaped.action.parameters["level_singularity_escape"] is True
    assert escaped.action.parameters["gripper_remains_closed"] is True
    original_target = np.asarray(first.action.parameters["T_base_tcp_live_target"])
    escape_target = np.asarray(escaped.action.parameters["T_base_tcp_target"])
    np.testing.assert_allclose(escape_target[:3, :3], original_target[:3, :3])
    np.testing.assert_allclose(escape_target[:2, 3], original_target[:2, 3])
    assert escape_target[2, 3] == pytest.approx(original_target[2, 3] + 0.050)

    # The recovery is deliberately one-shot.  If the arm still cannot move,
    # subsequent decisions remain bounded Servo commands until the ordinary
    # state timeout rather than dispatching repeated coarse plans.
    after_escape = controller.step(stalled_observation(1.3))
    assert after_escape.action.kind is ActionKind.SERVO_TCP
    assert after_escape.action.parameters["gripper_remains_closed"] is True


def test_level_progress_resets_stall_window_before_escape() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(
            level_stall_confirmation_windows=2,
            level_stall_min_progress_deg=1.0,
        ),
    )
    controller.state = TubeState.LEVEL_HELD_TUBE
    contact = TactileFeedback(
        valid=True,
        observed_at_s=0.99,
        contact_present=True,
        bilateral_contact=True,
        bilateral_contact_interior=True,
    )
    poses = (
        _transform(rotation_deg_xyz=(30.0, 0.0, 0.0)),
        _transform(rotation_deg_xyz=(30.0, 0.0, 0.0)),
        _transform(rotation_deg_xyz=(35.0, 0.0, 0.0)),
        _transform(rotation_deg_xyz=(35.0, 0.0, 0.0)),
    )
    decisions = []
    for index, pose in enumerate(poses):
        now_s = 1.0 + index * 0.1
        decisions.append(
            controller.step(
                _observation(
                    now_s,
                    tcp=pose,
                    tube_pose=_pose("tube", now_s, pose),
                    motion_complete=False,
                    tactile=replace(contact, observed_at_s=now_s - 0.01),
                )
            )
        )

    assert [item.action.kind for item in decisions] == [ActionKind.SERVO_TCP] * 4


def test_level_state_ignores_motion_correlated_slip_with_bilateral_interior_contact() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.LEVEL_HELD_TUBE
    tilted = _transform(rotation_deg_xyz=(30.0, 0.0, 0.0))
    decision = controller.step(
        _observation(
            1.0,
            tcp=tilted,
            tube_pose=_pose("tube", 1.0, tilted),
            motion_complete=False,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=0.99,
                contact_present=True,
                bilateral_contact=True,
                bilateral_contact_interior=True,
                slip_detected=True,
            ),
        )
    )

    assert decision.next_state is TubeState.LEVEL_HELD_TUBE
    assert decision.action.kind is ActionKind.SERVO_TCP
    assert decision.action.parameters["motion_correlated_slip_ignored"] is True


def test_level_state_requires_persistent_stationary_slip_before_hold() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(level_stationary_slip_confirmation_windows=2),
    )
    controller.state = TubeState.LEVEL_HELD_TUBE
    tilted = _transform(rotation_deg_xyz=(30.0, 0.0, 0.0))
    tactile = TactileFeedback(
        valid=True,
        observed_at_s=0.99,
        contact_present=True,
        bilateral_contact=True,
        bilateral_contact_interior=True,
        slip_detected=True,
    )

    first = controller.step(
        _observation(
            1.0,
            tcp=tilted,
            tube_pose=_pose("tube", 1.0, tilted),
            tactile=tactile,
        )
    )
    assert first.next_state is TubeState.LEVEL_HELD_TUBE
    assert first.action.kind is ActionKind.COLLECT_EVIDENCE
    assert first.action.parameters["stationary_slip_confirmation_windows"] == 1

    second = controller.step(
        _observation(
            1.1,
            tcp=tilted,
            tube_pose=_pose("tube", 1.1, tilted),
            tactile=replace(tactile, observed_at_s=1.09),
        )
    )
    assert second.next_state is TubeState.HOLD
    assert second.action.kind is ActionKind.STOP_AND_HOLD
    assert second.action.parameters["stationary_slip_confirmation_windows"] == 2


def test_level_state_noninterior_slip_still_holds_immediately() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.LEVEL_HELD_TUBE
    tilted = _transform(rotation_deg_xyz=(30.0, 0.0, 0.0))
    decision = controller.step(
        _observation(
            1.0,
            tcp=tilted,
            tube_pose=_pose("tube", 1.0, tilted),
            motion_complete=False,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=0.99,
                contact_present=True,
                bilateral_contact=True,
                bilateral_contact_interior=False,
                slip_detected=True,
            ),
        )
    )

    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD


def test_level_state_stops_before_motion_when_horizontal_direction_is_unobservable() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.LEVEL_HELD_TUBE
    decision = controller.step(
        _observation(
            1.0,
            tube_pose=_pose("tube", 1.0, np.eye(4)),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=0.99,
                contact_present=True,
                bilateral_contact=True,
            ),
        )
    )

    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD
    assert "held_tube_horizontal_direction_unobservable" in decision.action.reason


def test_translation_axis_fallback_cannot_authorize_robot_approach() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    fallback = _backend(qualified=False, fallback=True)
    observation = _observation(
        1.0,
        tube_pose=_pose("tube", 1.0, backend=fallback),
        socket_pose=_pose("socket", 1.0),
    )

    decision = controller.step(observation)

    assert decision.next_state is TubeState.OBSERVE
    assert decision.action.kind is ActionKind.COLLECT_EVIDENCE
    assert not decision.terminal
    assert "missing_full_se3" in decision.action.reason


def test_observe_allows_slow_semantic_bootstrap_without_relaxing_servo_pose_age() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(
            pose_max_age_s=0.2,
            observe_pose_max_age_s=4.0,
            observe_timeout_s=10.0,
        ),
    )
    tube = _pose("tube", 1.0)
    socket = _pose("socket", 1.0)

    initial = controller.step(
        _observation(3.0, tube_pose=tube, socket_pose=socket)
    )

    assert initial.action.kind is ActionKind.MOVE_TCP
    assert initial.next_state is TubeState.APPROACH_TUBE

    stale_during_servo = controller.step(
        _observation(3.1, tube_pose=tube, socket_pose=socket)
    )
    assert stale_during_servo.action.kind is ActionKind.PAUSE_ARM
    assert "pose_stale" in stale_during_servo.action.reason


def test_blocking_coarse_move_does_not_consume_next_state_timeout() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(default_stage_timeout_s=12.0),
    )
    initial = controller.step(
        _observation(
            1.0,
            tube_pose=_pose("tube", 1.0),
            socket_pose=_pose("socket", 1.0),
        )
    )
    assert initial.action.kind is ActionKind.MOVE_TCP
    assert initial.next_state is TubeState.APPROACH_TUBE

    # The resident MOVE_TCP dispatch may block for longer than the 12 s
    # evidence timeout.  Its first post-motion observation starts that timeout.
    target = np.asarray(initial.action.parameters["T_base_tcp_target"])
    after_slow_move = controller.step(
        _observation(
            31.0,
            tcp=target,
            tube_pose=_pose("tube", 31.0),
            socket_pose=_pose("socket", 31.0),
        )
    )
    assert after_slow_move.next_state is not TubeState.HOLD


def test_slow_approach_pauses_reacquires_and_resumes_with_finite_retry_budget() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(
            approach_timeout_s=60.0,
            max_approach_timeout_recoveries=2,
        ),
    )
    initial = controller.step(
        _observation(1.0, tube_pose=_pose("tube", 1.0), socket_pose=_pose("socket", 1.0))
    )
    target = np.asarray(initial.action.parameters["T_base_tcp_target"])
    measured = target.copy()
    measured[0, 3] += 0.02
    # The blocking coarse action has finished. Missing vision pauses motion,
    # and the old 12 s budget must not end a slow correction.
    controller.step(_observation(31.0, tcp=measured))
    after_old_timeout = controller.step(_observation(45.0, tcp=measured))
    assert after_old_timeout.action.kind is ActionKind.PAUSE_ARM
    assert not after_old_timeout.terminal

    for retry, now_s in enumerate((92.0, 153.0), start=1):
        paused = controller.step(_observation(now_s, tcp=measured))
        assert paused.action.kind is ActionKind.PAUSE_ARM
        assert paused.action.parameters["zero_active_servo"] is True
        assert f"automatic retry {retry}/2" in paused.action.reason
        assert not paused.terminal
        resumed = controller.step(
            _observation(now_s + 0.1, tcp=measured, tube_pose=_pose("tube", now_s + 0.1))
        )
        assert resumed.action.kind is ActionKind.SERVO_TCP
        assert resumed.next_state is TubeState.APPROACH_TUBE
        # Retry must not rebase the target-drift reference.
        assert np.allclose(controller._initial_target, target)

    exhausted = controller.step(_observation(214.0, tcp=measured))
    assert exhausted.action.kind is ActionKind.STOP_AND_HOLD
    assert exhausted.terminal


def test_approach_retry_never_masks_hardware_fault() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(approach_timeout_s=60.0, max_approach_timeout_recoveries=2),
    )
    controller.state = TubeState.APPROACH_TUBE
    controller.step(_observation(1.0))
    observation = _observation(62.0)
    observation = replace(
        observation, robot=replace(observation.robot, fault="controller_reflex")
    )

    stopped = controller.step(observation)

    assert stopped.action.kind is ActionKind.STOP_AND_HOLD
    assert "controller_reflex" in stopped.action.reason
    assert controller._approach_timeout_recoveries == 0


def test_background_dominated_mask_cannot_be_used_as_metric_tube_pose() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    tube = ObjectPoseEstimate(
        object_id="tube",
        T_base_object=np.eye(4),
        observed_at_s=0.99,
        position_std_m=0.001,
        orientation_std_deg=1.0,
        backend=_backend(),
        tracking_id="tube-background-dominated",
        inference_latency_s=0.01,
        mask_id="sam3-mask",
        mask_iou=0.9,
        valid_depth_count=2060,
        foreground_depth_support_count=55,
        foreground_depth_support_fraction=55 / 2060,
        background_depth_fraction=1813 / 2060,
        masked_depth_cluster_qualified=False,
        axis_sign_source="open_end_detector",
        cylindrical_roll_canonicalized=True,
    )

    decision = controller.step(
        _observation(1.0, tube_pose=tube, socket_pose=_pose("socket", 1.0))
    )

    assert decision.next_state is TubeState.OBSERVE
    assert decision.action.kind is ActionKind.COLLECT_EVIDENCE
    assert "background_depth_dominates_mask" in decision.action.reason


def test_tracking_identity_change_fails_closed_before_servo() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.step(
        _observation(
            1.0,
            tube_pose=_pose("tube", 1.0),
            socket_pose=_pose("socket", 1.0),
        )
    )
    changed = _pose("tube", 1.1)
    object.__setattr__(changed, "tracking_id", "different-tube-track")

    decision = controller.step(
        _observation(
            1.1,
            tube_pose=changed,
            socket_pose=_pose("socket", 1.1),
        )
    )

    assert decision.next_state is TubeState.APPROACH_TUBE
    assert decision.action.kind is ActionKind.PAUSE_ARM
    assert not decision.terminal
    assert "tube_tracking_id_changed" in decision.action.reason


def test_continuous_close_is_issued_once_then_tactile_stops_it() -> None:
    policy = ClosedLoopPolicy(servo_settle_windows=2)
    controller = TubeClosedLoopController(geometry=_geometry(), policy=policy)
    controller.state = TubeState.SIDE_GRASP
    tube_1 = _pose("tube", 1.0)
    tube_2 = _pose("tube", 1.1)

    first = controller.step(_observation(1.0, tube_pose=tube_1))
    assert first.action.kind is ActionKind.SERVO_TCP
    second = controller.step(_observation(1.1, tube_pose=tube_2))
    assert second.action.kind is ActionKind.START_CONTINUOUS_CLOSE
    assert second.action.parameters["command_count"] == 1
    assert (
        second.action.parameters["stop_predicate"]
        == "first_fresh_optical_pad_contact"
    )
    assert second.action.parameters["hande_object_status_role"] == "corroborating_only"

    monitoring = controller.step(
        _observation(1.2, tube_pose=_pose("tube", 1.2), moving=True, width_m=0.025)
    )
    assert monitoring.action.kind is ActionKind.MONITOR

    bilateral = TactileFeedback(
        valid=True,
        observed_at_s=1.29,
        contact_present=True,
        bilateral_contact=True,
        bilateral_contact_interior=True,
    )
    stopping = controller.step(
        _observation(
            1.3,
            tube_pose=_pose("tube", 1.3),
            moving=True,
            width_m=0.015,
            tactile=bilateral,
        )
    )
    assert stopping.action.kind is ActionKind.STOP_GRIPPER

    confirming = controller.step(
        _observation(
            1.4,
            tube_pose=_pose("tube", 1.4),
            moving=False,
            width_m=0.015,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.39,
                contact_present=True,
                bilateral_contact=True,
                bilateral_contact_interior=True,
            ),
        )
    )
    assert confirming.action.kind is ActionKind.COLLECT_EVIDENCE
    extracted = controller.step(
        _observation(
            1.5,
            tube_pose=_pose("tube", 1.5),
            moving=False,
            width_m=0.015,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.49,
                contact_present=True,
                bilateral_contact=True,
                bilateral_contact_interior=True,
            ),
        )
    )
    assert extracted.next_state is TubeState.EXTRACT
    assert extracted.action.kind is ActionKind.SERVO_TCP
    command = np.asarray(extracted.action.parameters["T_base_tcp_command"])
    assert np.linalg.norm(command[:3, 3]) == pytest.approx(
        controller.policy.servo_max_translation_step_m
    )


def test_hande_object_without_tactile_enters_bounded_deeper_retry() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.SIDE_GRASP
    controller._phase = "closing"
    controller._final_contact_start_tcp = _transform((0.0, 0.0, 0.050))
    controller._final_contact_target = np.eye(4)
    observation = _observation(
        1.0,
        moving=False,
        width_m=0.020,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.99,
            contact_present=False,
        ),
    )
    observation = replace(
        observation,
        gripper=replace(observation.gripper, object_detected=True),
    )

    decision = controller.step(observation)

    assert decision.next_state is TubeState.SIDE_GRASP
    assert decision.action.kind is ActionKind.STOP_GRIPPER
    first_window = controller.step(
        replace(
            observation,
            observation_id="obs-2",
            now_s=1.1,
            tactile=replace(observation.tactile, observed_at_s=1.09),
        )
    )
    assert first_window.action.kind is ActionKind.COLLECT_EVIDENCE
    retry = controller.step(
        replace(
            observation,
            observation_id="obs-3",
            now_s=1.2,
            tactile=replace(observation.tactile, observed_at_s=1.19),
        )
    )
    assert retry.action.kind is ActionKind.OPEN_GRIPPER_STEP
    assert retry.action.parameters["retry_step_m"] == pytest.approx(0.005)
    assert retry.action.parameters["retry_total_travel_m"] == pytest.approx(0.005)
    np.testing.assert_allclose(
        controller._shallow_retry_target[:3, 3],
        [0.0, 0.0, -0.005],
    )

    no_contact = lambda now: TactileFeedback(  # noqa: E731
        valid=True, observed_at_s=now - 0.01
    )
    edge = lambda now: TactileFeedback(  # noqa: E731
        valid=True,
        observed_at_s=now - 0.01,
        contact_present=True,
        bilateral_contact=True,
    )
    interior = lambda now: TactileFeedback(  # noqa: E731
        valid=True,
        observed_at_s=now - 0.01,
        contact_present=True,
        bilateral_contact=True,
        bilateral_contact_interior=True,
    )

    first_probe = controller.step(
        _observation(1.3, width_m=0.0495, tactile=no_contact(1.3))
    )
    assert first_probe.action.kind is ActionKind.SERVO_TCP
    first_target = np.asarray(first_probe.action.parameters["T_base_tcp_command"])
    first_reclose = controller.step(
        _observation(1.4, tcp=first_target, width_m=0.0495, tactile=no_contact(1.4))
    )
    assert first_reclose.action.kind is ActionKind.START_CONTINUOUS_CLOSE
    assert first_reclose.action.parameters["shallow_contact_retry"] is True
    assert first_reclose.action.parameters["command_count"] == 1

    stopped_at_edge = controller.step(
        _observation(1.5, tcp=first_target, moving=True, width_m=0.018, tactile=edge(1.5))
    )
    assert stopped_at_edge.action.kind is ActionKind.STOP_GRIPPER
    assert controller.step(
        _observation(1.6, tcp=first_target, width_m=0.018, tactile=edge(1.6))
    ).action.kind is ActionKind.COLLECT_EVIDENCE
    second_retry = controller.step(
        _observation(1.7, tcp=first_target, width_m=0.018, tactile=edge(1.7))
    )
    assert second_retry.action.kind is ActionKind.OPEN_GRIPPER_STEP
    assert second_retry.action.parameters["retry_total_travel_m"] == pytest.approx(0.010)

    second_probe = controller.step(
        _observation(1.8, tcp=first_target, width_m=0.0495, tactile=no_contact(1.8))
    )
    second_target = np.asarray(second_probe.action.parameters["T_base_tcp_command"])
    np.testing.assert_allclose(second_target[:3, 3], [0.0, 0.0, -0.010])
    assert controller.step(
        _observation(1.9, tcp=second_target, width_m=0.0495, tactile=no_contact(1.9))
    ).action.kind is ActionKind.START_CONTINUOUS_CLOSE
    assert controller.step(
        _observation(
            2.0,
            tcp=second_target,
            moving=True,
            width_m=0.018,
            tactile=interior(2.0),
        )
    ).action.kind is ActionKind.STOP_GRIPPER
    confirming = controller.step(
        _observation(2.1, tcp=second_target, width_m=0.018, tactile=interior(2.1))
    )
    assert confirming.action.kind is ActionKind.COLLECT_EVIDENCE
    extracted = controller.step(
        _observation(2.2, tcp=second_target, width_m=0.018, tactile=interior(2.2))
    )
    assert extracted.next_state is TubeState.EXTRACT
    assert extracted.action.kind is ActionKind.SERVO_TCP


def test_shallow_retry_never_exceeds_two_steps_or_ten_millimetres() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.SIDE_GRASP
    controller._phase = "close_stopping"
    controller._final_contact_start_tcp = _transform((0.0, 0.0, 0.050))
    controller._final_contact_target = np.eye(4)
    controller._shallow_contact_retries = 2
    controller._shallow_retry_total_travel_m = 0.010
    controller._shallow_contact_unready_windows = 1

    decision = controller.step(
        _observation(
            1.0,
            width_m=0.018,
            tactile=TactileFeedback(valid=True, observed_at_s=0.99),
        )
    )

    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD
    assert "retry budget exhausted" in decision.action.reason


def test_stationary_close_without_tactile_stops_before_stationary_confirmation() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.SIDE_GRASP
    controller._phase = "closing"

    decision = controller.step(
        _observation(
            1.0,
            moving=False,
            width_m=0.020,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=0.99,
                contact_present=False,
            ),
        )
    )

    assert decision.next_state is TubeState.SIDE_GRASP
    assert decision.action.kind is ActionKind.STOP_GRIPPER
    assert "collect stationary pad images" in decision.action.reason


def test_near_gripper_occlusion_freezes_pose_and_switches_to_tactile() -> None:
    policy = ClosedLoopPolicy(servo_settle_windows=2)
    controller = TubeClosedLoopController(geometry=_geometry(), policy=policy)
    controller.state = TubeState.SIDE_GRASP

    contact = TactileFeedback(
        valid=True,
        observed_at_s=1.09,
        contact_present=True,
    )
    controller.step(_observation(1.0, tube_pose=_pose("tube", 1.0)))
    occluded = controller.step(
        _observation(
            1.1,
            tube_pose=_pose("tube", 1.1, occluded=True),
            tactile=contact,
        )
    )

    assert occluded.action.kind is ActionKind.START_CONTINUOUS_CLOSE
    assert controller.state is TubeState.SIDE_GRASP


@pytest.mark.parametrize("explicit_occlusion", (False, True))
def test_converged_occlusion_allows_first_close_without_preexisting_contact(
    explicit_occlusion: bool,
) -> None:
    policy = ClosedLoopPolicy(servo_settle_windows=2, pose_max_age_s=0.2)
    controller = TubeClosedLoopController(geometry=_geometry(), policy=policy)
    controller.state = TubeState.SIDE_GRASP
    controller.step(_observation(1.0, tube_pose=_pose("tube", 1.0)))

    decision = controller.step(
        _observation(
            1.1,
            tube_pose=_pose("tube", 1.1, occluded=True) if explicit_occlusion else None,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.09,
                contact_present=False,
            ),
        )
    )

    assert decision.action.kind is ActionKind.START_CONTINUOUS_CLOSE
    assert decision.action.parameters["zero_active_servo"] is True
    assert controller._vision_frozen_for_close is True


@pytest.mark.parametrize(
    ("now_s", "contact"),
    [
        (1.3, False),
        (1.3, True),
    ],
)
def test_missing_pose_does_not_switch_with_stale_cache(
    now_s: float,
    contact: bool,
) -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(pose_max_age_s=0.2),
    )
    controller.state = TubeState.SIDE_GRASP
    controller.step(_observation(1.0, tube_pose=_pose("tube", 1.0)))

    decision = controller.step(
        _observation(
            now_s,
            tube_pose=None,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=now_s - 0.01,
                contact_present=contact,
            ),
        )
    )

    assert decision.action.kind is ActionKind.PAUSE_ARM
    assert "required_live_pose_missing" in decision.action.reason


@pytest.mark.parametrize(
    ("position_error_m", "motion_complete", "tactile_valid"),
    ((0.004, True, True), (0.0, False, True), (0.0, True, False)),
)
def test_cached_occlusion_never_authorizes_blind_descent_or_unobserved_close(
    position_error_m: float, motion_complete: bool, tactile_valid: bool,
) -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(), policy=ClosedLoopPolicy(pose_max_age_s=0.2),
    )
    controller.state = TubeState.SIDE_GRASP
    controller.step(_observation(1.0, tube_pose=_pose("tube", 1.0)))
    measured = _transform((0.0, 0.0, position_error_m))

    decision = controller.step(
        _observation(
            1.1,
            tcp=measured,
            tube_pose=None,
            motion_complete=motion_complete,
            tactile=TactileFeedback(valid=tactile_valid, observed_at_s=1.09),
        )
    )

    assert decision.action.kind is ActionKind.PAUSE_ARM
    assert decision.action.parameters["zero_active_servo"] is True


def test_occluded_changed_tracking_identity_cannot_authorize_close() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.SIDE_GRASP
    controller.step(_observation(1.0, tube_pose=_pose("tube", 1.0)))
    switched = replace(_pose("tube", 1.1, occluded=True), tracking_id="new-tube")

    decision = controller.step(_observation(1.1, tube_pose=switched))

    assert decision.action.kind is ActionKind.PAUSE_ARM
    assert "tube_tracking_id_changed" in decision.action.reason


@pytest.mark.parametrize("explicit_occlusion", (False, True))
def test_task_occlusion_window_closes_in_place_instead_of_blind_final_descent(
    explicit_occlusion: bool,
) -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(occluded_grasp_close_distance_m=0.008),
    )
    controller.state = TubeState.SIDE_GRASP
    measured = _transform((0.0, 0.0, 0.0076))
    visible = controller.step(
        _observation(1.0, tcp=measured, tube_pose=_pose("tube", 1.0))
    )
    assert visible.action.kind is ActionKind.SERVO_TCP

    occluded = controller.step(_observation(
        1.1,
        tcp=measured,
        motion_complete=False,
        tube_pose=_pose("tube", 1.1, occluded=True) if explicit_occlusion else None,
    ))

    assert occluded.action.kind is ActionKind.START_CONTINUOUS_CLOSE
    assert occluded.action.parameters["zero_active_servo"] is True
    assert "T_base_tcp_command" not in occluded.action.parameters
    assert controller._phase == "closing"
    monitored = controller.step(_observation(1.2, tcp=measured, moving=True))
    assert monitored.action.kind is ActionKind.MONITOR


@pytest.mark.parametrize(
    ("translation", "angle_deg", "now_s"),
    ((0.009, 0.0, 1.1), (0.0076, 2.1, 1.1), (0.0076, 0.0, 1.3)),
)
def test_task_occlusion_window_retains_distance_angle_and_cache_freshness(
    translation: float, angle_deg: float, now_s: float,
) -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(occluded_grasp_close_distance_m=0.008, pose_max_age_s=0.2),
    )
    controller.state = TubeState.SIDE_GRASP
    controller.step(_observation(1.0, tube_pose=_pose("tube", 1.0)))

    decision = controller.step(_observation(
        now_s, tcp=_transform((0.0, 0.0, translation), (0.0, 0.0, angle_deg)),
    ))

    assert decision.action.kind is ActionKind.PAUSE_ARM
    assert not decision.terminal


def test_task_resume_near_grasp_avoids_retreat_then_closes_on_occlusion() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(occluded_grasp_close_distance_m=0.008),
    )
    measured = _transform((0.0, 0.0, 0.0076))

    resumed = controller.step(_observation(
        1.0, tcp=measured, tube_pose=_pose("tube", 1.0), socket_pose=_pose("socket", 1.0),
    ))

    assert resumed.next_state is TubeState.SIDE_GRASP
    assert resumed.action.kind is ActionKind.PAUSE_ARM
    assert resumed.action.parameters["resume_grasp_in_place"] is True
    closed = controller.step(_observation(1.1, tcp=measured))
    assert closed.action.kind is ActionKind.START_CONTINUOUS_CLOSE


def test_task_resume_needs_recent_frames_even_when_observe_age_budget_allows_them() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(occluded_grasp_close_distance_m=0.008),
    )

    decision = controller.step(_observation(
        2.0, tube_pose=_pose("tube", 1.0), socket_pose=_pose("socket", 1.0),
    ))

    assert decision.next_state is TubeState.OBSERVE
    assert decision.action.kind is ActionKind.COLLECT_EVIDENCE


def test_unknown_pad_sign_uses_bounded_probe_then_one_reversal() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.SIDE_GRASP
    controller._phase = "close_stopping"
    controller._contact_stop_width_m = 0.020
    controller._single_contact_windows = 1
    single = TactileFeedback(
        valid=True,
        observed_at_s=0.99,
        contact_present=True,
        single_contact_side="left",
    )

    first_probe = controller.step(
        _observation(1.0, width_m=0.020, tactile=single)
    )
    assert first_probe.action.kind is ActionKind.OPEN_AND_RECENTER
    assert first_probe.action.parameters["tcp_local_x_offset_m"] == pytest.approx(0.001)
    assert first_probe.action.parameters["strategy"] == "bounded_unknown_sensor_sign_probe"

    # Once the bounded probe has completed, visual re-grounding must preserve
    # its accumulated jaw-axis offset rather than commanding the TCP back to
    # the original one-pad-contact pose.
    controller._phase = "recenter_complete"
    shifted_tcp = _transform((0.001, 0.0, 0.0))
    after_probe = controller.step(
        _observation(
            1.05,
            tcp=shifted_tcp,
            tube_pose=_pose("tube", 1.05),
        )
    )
    np.testing.assert_allclose(
        np.asarray(after_probe.action.parameters["T_base_tcp_live_target"])[:3, 3],
        [0.001, 0.0, 0.0],
    )

    controller._phase = "close_stopping"
    controller._contact_stop_width_m = 0.021
    controller._single_contact_windows = 1
    reversed_probe = controller.step(
        _observation(
            1.1,
            width_m=0.021,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.09,
                contact_present=True,
                single_contact_side="left",
            ),
        )
    )
    assert reversed_probe.action.kind is ActionKind.OPEN_AND_RECENTER
    assert reversed_probe.action.parameters["tcp_local_x_offset_m"] == pytest.approx(-0.002)
    assert reversed_probe.action.parameters["direction_reversed"] is True


def test_one_sided_contact_fails_closed_without_bounded_tcp_servo() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.SIDE_GRASP
    controller._phase = "close_stopping"
    controller._contact_stop_width_m = 0.020
    controller._single_contact_windows = 1
    observation = _observation(
        1.0,
        width_m=0.020,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.99,
            contact_present=True,
            single_contact_side="right",
        ),
        bounded_servo_supported=False,
    )

    decision = controller.step(observation)

    assert decision.next_state is TubeState.HOLD
    assert "bounded TCP jaw-axis servo" in decision.action.reason


def test_tactile_angle_reclamps_and_qualified_jam_retracts() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.PIVOT_EVALUATE
    controller._pivot_steps = 1
    pivot = controller.step(
        _observation(
            1.0,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=0.99,
                contact_present=True,
                bilateral_contact=True,
                relative_angle_target_reached=True,
            ),
        )
    )
    assert pivot.next_state is TubeState.RECLAMP
    assert pivot.action.kind is ActionKind.RECLAMP

    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.INSERT_EVALUATE
    jam = controller.step(
        _observation(
            2.0,
            socket_pose=_pose("socket", 2.0),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.99,
                contact_present=True,
                bilateral_contact=True,
                insertion_jam=True,
                insertion_resistance_high=True,
            ),
            insertion_progress_m=0.0001,
            insertion_depth_m=0.005,
        )
    )
    assert jam.next_state is TubeState.JAM_RECOVERY_RETRACT
    assert jam.action.kind is ActionKind.RETRACT_INSERTION


def test_debounced_tactile_seat_completes_insertion_without_target_depth() -> None:
    policy = ClosedLoopPolicy(
        insertion_target_depth_m=None,
        insertion_max_travel_m=0.030,
    )
    controller = TubeClosedLoopController(geometry=_geometry(), policy=policy)
    controller.state = TubeState.INSERT_EVALUATE

    decision = controller.step(
        _observation(
            2.0,
            socket_pose=_pose("socket", 2.0),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.99,
                contact_present=True,
                bilateral_contact=True,
                insertion_resistance_high=True,
                insertion_seated=True,
            ),
            insertion_progress_m=0.0001,
            insertion_depth_m=0.006,
        )
    )

    assert decision.next_state is TubeState.RELEASE
    assert decision.action.kind is ActionKind.COLLECT_EVIDENCE
    assert "insertion_seated" in decision.predicates


def test_event_driven_insertion_stops_at_maximum_travel_without_seat() -> None:
    policy = ClosedLoopPolicy(
        insertion_target_depth_m=None,
        insertion_max_travel_m=0.009,
    )
    controller = TubeClosedLoopController(geometry=_geometry(), policy=policy)
    controller.state = TubeState.INSERT_EVALUATE

    decision = controller.step(
        _observation(
            2.0,
            socket_pose=_pose("socket", 2.0),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.99,
                contact_present=True,
                bilateral_contact=True,
            ),
            insertion_progress_m=0.0015,
            insertion_depth_m=0.009,
        )
    )

    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD
    assert "maximum bounded insertion travel" in decision.action.reason


def test_event_driven_insertion_never_commands_a_step_past_maximum_travel() -> None:
    policy = ClosedLoopPolicy(
        insertion_step_m=0.0015,
        insertion_target_depth_m=None,
        insertion_max_travel_m=0.010,
    )
    controller = TubeClosedLoopController(geometry=_geometry(), policy=policy)
    controller.state = TubeState.INSERT_EVALUATE
    controller._insertion_depth_commanded_m = 0.009

    decision = controller.step(
        _observation(
            2.0,
            socket_pose=_pose("socket", 2.0),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.99,
                contact_present=True,
                bilateral_contact=True,
            ),
            insertion_progress_m=0.0015,
            insertion_depth_m=0.009,
        )
    )

    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD
    assert "smaller than one microstep" in decision.action.reason


def test_observe_waits_for_session_tactile_baseline_before_robot_motion() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    observation = _observation(
        1.0,
        tube_pose=_pose("tube", 1.0),
        socket_pose=_pose("socket", 1.0),
        tactile=TactileFeedback(valid=False, observed_at_s=0.99),
    )

    decision = controller.step(observation)

    assert decision.next_state is TubeState.OBSERVE
    assert decision.action.kind is ActionKind.COLLECT_EVIDENCE
    assert "tactile_evidence_invalid" in decision.action.reason


def test_align_waits_for_session_insertion_reference_before_first_push() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.ALIGN_SOCKET
    controller._T_tube_tcp_held = np.eye(4)
    observation = _observation(
        1.0,
        tube_pose=_pose("tube", 1.0),
        socket_pose=_pose("socket", 1.0),
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.99,
            contact_present=True,
            bilateral_contact=True,
            insertion_reference_ready=False,
        ),
        tube_socket_axes_aligned=True,
        tube_tip_at_socket_entry=True,
    )

    decision = controller.step(observation)

    assert decision.next_state is TubeState.ALIGN_SOCKET
    assert decision.action.kind is ActionKind.SERVO_TCP
    assert "tube_socket_axes_aligned" not in decision.predicates
    assert "tube_tip_at_socket_entry" not in decision.predicates


@pytest.mark.parametrize(
    ("tactile", "reason"),
    (
        (
            TactileFeedback(
                valid=True,
                observed_at_s=1.99,
                contact_present=True,
                bilateral_contact=True,
                insertion_reference_ready=False,
            ),
            "insertion tactile reference",
        ),
        (
            TactileFeedback(
                valid=True,
                observed_at_s=1.99,
                contact_present=True,
                bilateral_contact=False,
                insertion_reference_ready=True,
            ),
            "bilateral tactile contact",
        ),
    ),
    ids=("reference-lost", "bilateral-contact-lost"),
)
def test_insertion_holds_if_session_tactile_reference_or_contact_is_lost(
    tactile: TactileFeedback,
    reason: str,
) -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.INSERT_EVALUATE

    decision = controller.step(
        _observation(
            2.0,
            socket_pose=_pose("socket", 2.0),
            tactile=tactile,
            insertion_progress_m=0.0015,
            insertion_depth_m=0.006,
        )
    )

    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD
    assert reason in decision.action.reason


@pytest.mark.parametrize(
    "tactile",
    (
        TactileFeedback(
            valid=False,
            observed_at_s=0.99,
            contact_present=True,
            bilateral_contact=True,
        ),
        TactileFeedback(
            valid=True,
            observed_at_s=0.0,
            contact_present=True,
            bilateral_contact=True,
        ),
    ),
    ids=("invalid", "stale"),
)
def test_extract_cannot_consume_unqualified_contact(tactile: TactileFeedback) -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.EXTRACT
    controller._extract_target = np.eye(4)

    decision = controller.step(
        _observation(
            1.0,
            tactile=tactile,
            tube_clear_of_source=True,
        )
    )

    assert decision.next_state is TubeState.HOLD
    assert decision.action.kind is ActionKind.STOP_AND_HOLD


def test_contact_loss_between_extract_servo_steps_stops_immediately() -> None:
    controller = TubeClosedLoopController(
        geometry=_geometry(),
        policy=ClosedLoopPolicy(shallow_contact_confirmation_windows=1),
    )
    controller.state = TubeState.SIDE_GRASP
    controller._phase = "close_stopping"
    controller._final_contact_start_tcp = _transform((0.0, 0.0, 0.050))
    controller._final_contact_target = np.eye(4)
    started = controller.step(
        _observation(
            1.0,
            width_m=0.015,
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=0.99,
                contact_present=True,
                bilateral_contact=True,
                bilateral_contact_interior=True,
            ),
        )
    )
    assert started.next_state is TubeState.EXTRACT
    assert started.action.kind is ActionKind.SERVO_TCP

    stopped = controller.step(
        _observation(
            1.1,
            tcp=np.asarray(started.action.parameters["T_base_tcp_command"]),
            tactile=TactileFeedback(valid=True, observed_at_s=1.09),
        )
    )
    assert stopped.next_state is TubeState.HOLD
    assert stopped.action.kind is ActionKind.STOP_AND_HOLD
    assert "contact was lost during extraction" in stopped.action.reason


def test_reclamp_retries_held_relation_before_approach_state_transition() -> None:
    controller = TubeClosedLoopController(geometry=_geometry())
    controller.state = TubeState.RECLAMP
    stable = TactileFeedback(
        valid=True,
        observed_at_s=0.99,
        contact_present=True,
        bilateral_contact=True,
        rotation_stopped=True,
    )

    missing = controller.step(
        _observation(
            1.0,
            tube_pose=None,
            socket_pose=_pose("socket", 1.0),
            tactile=stable,
        )
    )
    assert missing.next_state is TubeState.RECLAMP
    assert missing.action.kind is ActionKind.PAUSE_ARM
    assert controller._T_tube_tcp_held is None

    recovered = controller.step(
        _observation(
            1.1,
            tube_pose=_pose("tube", 1.1),
            socket_pose=_pose("socket", 1.1),
            tactile=TactileFeedback(
                valid=True,
                observed_at_s=1.09,
                contact_present=True,
                bilateral_contact=True,
                rotation_stopped=True,
            ),
        )
    )
    assert recovered.next_state is TubeState.APPROACH_SOCKET
    assert recovered.action.kind is ActionKind.MOVE_TCP
    assert controller._T_tube_tcp_held is not None


def test_nominal_trace_runs_all_seven_stages_without_operator_transitions() -> None:
    policy = ClosedLoopPolicy(
        servo_settle_windows=1,
        shallow_contact_confirmation_windows=1,
        level_settle_windows=1,
        insertion_target_depth_m=0.003,
        default_stage_timeout_s=100.0,
        close_timeout_s=100.0,
        insertion_timeout_s=100.0,
    )
    controller = TubeClosedLoopController(geometry=_geometry(), policy=policy)
    tube = _transform((0.50, 0.00, 0.10), (20.0, 0.0, 0.0))
    socket = np.eye(4)
    socket[:3, 3] = [0.70, 0.00, 0.10]
    tcp = np.eye(4)
    tcp[:3, 3] = [0.30, 0.00, 0.30]
    decisions = []

    def step(
        timestamp: float,
        *,
        tcp_pose: np.ndarray,
        tube_transform: np.ndarray,
        tactile: TactileFeedback | None = None,
        **flags,
    ):
        decision = controller.step(
            _observation(
                timestamp,
                tcp=tcp_pose,
                tube_pose=_pose("tube", timestamp, tube_transform),
                socket_pose=_pose("socket", timestamp, socket),
                tactile=tactile,
                **flags,
            )
        )
        decisions.append(decision)
        return decision

    decision = step(0.1, tcp_pose=tcp, tube_transform=tube)
    top = np.asarray(decision.action.parameters["T_base_tcp_target"])
    decision = step(0.2, tcp_pose=top, tube_transform=tube)
    grasp = np.asarray(decision.action.parameters["T_base_tcp_live_target"])
    step(0.3, tcp_pose=grasp, tube_transform=tube)
    step(0.4, tcp_pose=grasp, tube_transform=tube, moving=True, width_m=0.030)
    contact = TactileFeedback(
        valid=True,
        observed_at_s=0.49,
        contact_present=True,
        bilateral_contact=True,
        bilateral_contact_interior=True,
    )
    step(
        0.5,
        tcp_pose=grasp,
        tube_transform=tube,
        moving=True,
        width_m=0.015,
        tactile=contact,
    )
    contact = TactileFeedback(
        valid=True,
        observed_at_s=0.59,
        contact_present=True,
        bilateral_contact=True,
        bilateral_contact_interior=True,
    )
    decision = step(
        0.6,
        tcp_pose=grasp,
        tube_transform=tube,
        width_m=0.015,
        tactile=contact,
    )
    assert decision.action.kind is ActionKind.SERVO_TCP
    extracted_tcp = np.asarray(decision.action.parameters["T_base_tcp_live_target"])
    extracted_tube = extracted_tcp.copy()
    decision = step(
        0.7,
        tcp_pose=extracted_tcp,
        tube_transform=extracted_tube,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.69,
            contact_present=True,
            bilateral_contact=True,
        ),
        tube_clear_of_source=True,
    )
    assert decision.next_state is TubeState.LEVEL_HELD_TUBE
    assert decision.action.kind is ActionKind.SERVO_TCP
    level_tcp = np.asarray(decision.action.parameters["T_base_tcp_live_target"])
    level_tube = level_tcp.copy()
    decision = step(
        0.75,
        tcp_pose=level_tcp,
        tube_transform=level_tube,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.74,
            contact_present=True,
            bilateral_contact=True,
        ),
    )
    assert decision.next_state is TubeState.PIVOT_MICRO_LOOSEN
    assert decision.action.kind is ActionKind.OPEN_GRIPPER_STEP
    step(
        0.8,
        tcp_pose=level_tcp,
        tube_transform=level_tube,
        width_m=0.016,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.79,
            contact_present=True,
            bilateral_contact=True,
        ),
    )
    step(
        0.9,
        tcp_pose=level_tcp,
        tube_transform=level_tube,
        width_m=0.016,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.89,
            contact_present=True,
            bilateral_contact=True,
            relative_angle_target_reached=True,
        ),
    )
    decision = step(
        1.0,
        tcp_pose=level_tcp,
        tube_transform=level_tube,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=0.99,
            contact_present=True,
            bilateral_contact=True,
            relative_angle_target_reached=True,
            rotation_stopped=True,
        ),
    )
    rack_standoff = np.asarray(decision.action.parameters["T_base_tcp_target"])
    decision = step(
        1.1,
        tcp_pose=rack_standoff,
        tube_transform=rack_standoff,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=1.09,
            contact_present=True,
            bilateral_contact=True,
        ),
    )
    rack_entry = np.asarray(decision.action.parameters["T_base_tcp_live_target"])
    step(
        1.2,
        tcp_pose=rack_entry,
        tube_transform=rack_entry,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=1.19,
            contact_present=True,
            bilateral_contact=True,
        ),
        tube_tip_at_socket_entry=True,
        tube_socket_axes_aligned=True,
    )
    step(
        1.3,
        tcp_pose=rack_entry,
        tube_transform=rack_entry,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=1.29,
            contact_present=True,
            bilateral_contact=True,
        ),
    )
    step(
        1.4,
        tcp_pose=rack_entry,
        tube_transform=rack_entry,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=1.39,
            contact_present=True,
            bilateral_contact=True,
        ),
        insertion_progress_m=0.0015,
        insertion_depth_m=0.0015,
    )
    step(
        1.5,
        tcp_pose=rack_entry,
        tube_transform=rack_entry,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=1.49,
            contact_present=True,
            bilateral_contact=True,
        ),
    )
    step(
        1.6,
        tcp_pose=rack_entry,
        tube_transform=rack_entry,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=1.59,
            contact_present=True,
            bilateral_contact=True,
        ),
        insertion_progress_m=0.0015,
        insertion_depth_m=0.003,
    )
    step(
        1.7,
        tcp_pose=rack_entry,
        tube_transform=rack_entry,
        width_m=0.015,
        tactile=TactileFeedback(
            valid=True,
            observed_at_s=1.69,
            contact_present=True,
            bilateral_contact=True,
        ),
        tube_stays_supported=True,
    )
    decision = step(
        1.8,
        tcp_pose=rack_entry,
        tube_transform=rack_entry,
        width_m=0.050,
        open_confirmed=True,
        tactile=TactileFeedback(valid=True, observed_at_s=1.79),
        tube_stays_supported=True,
    )
    retreat = np.asarray(decision.action.parameters["T_base_tcp_target"])
    step(
        1.9,
        tcp_pose=retreat,
        tube_transform=rack_entry,
        width_m=0.050,
        open_confirmed=True,
        tactile=TactileFeedback(valid=True, observed_at_s=1.89),
        tube_stays_supported=True,
    )

    assert controller.state is TubeState.COMPLETE
    visited = {decision.next_state for decision in decisions}
    assert {
        TubeState.APPROACH_TUBE,
        TubeState.SIDE_GRASP,
        TubeState.EXTRACT,
        TubeState.LEVEL_HELD_TUBE,
        TubeState.PIVOT_MICRO_LOOSEN,
        TubeState.PIVOT_EVALUATE,
        TubeState.RECLAMP,
        TubeState.APPROACH_SOCKET,
        TubeState.ALIGN_SOCKET,
        TubeState.INSERT_MICROSTEP,
        TubeState.INSERT_EVALUATE,
        TubeState.RELEASE,
        TubeState.RETREAT,
        TubeState.COMPLETE,
    } <= visited


def _load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_closed_loop.py"
    spec = importlib.util.spec_from_file_location("test_tube_closed_loop_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RunnerDecision:
    def __init__(self, action: object, *, terminal: bool = False) -> None:
        self.action = action
        self.terminal = terminal

    def to_dict(self) -> dict[str, object]:
        return {
            "action": {"kind": self.action.kind.value},
            "terminal": self.terminal,
        }


class _RunnerController:
    def __init__(
        self,
        cli: object,
        *,
        action_kind: ActionKind = ActionKind.COLLECT_EVIDENCE,
        terminal_state: TubeState | None = None,
    ) -> None:
        self._cli = cli
        self._action_kind = action_kind
        self._terminal_state = terminal_state
        self.state = TubeState.OBSERVE

    def step(self, _observation: object) -> _RunnerDecision:
        if self._terminal_state is not None:
            self.state = self._terminal_state
        action = self._cli.ClosedLoopAction(
            action_id="action-test",
            kind=self._action_kind,
            state=self.state,
            parameters={},
            reason="runner lifecycle fixture",
        )
        return _RunnerDecision(action, terminal=self._terminal_state is not None)


class _RunnerIO:
    def __init__(
        self,
        *,
        observe_error: BaseException | None = None,
        dispatch_error: Exception | None = None,
        hold_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.observe_error = observe_error
        self.dispatch_error = dispatch_error
        self.hold_error = hold_error
        self.close_error = close_error
        self.hold_reasons: list[str] = []
        self.closed = False

    def observe(self, *, state: TubeState) -> object:
        del state
        if self.observe_error is not None:
            raise self.observe_error
        return replace(_observation(1.0), observation_id="obs-runner")

    def dispatch(self, _action: object) -> dict[str, object]:
        if self.dispatch_error is not None:
            raise self.dispatch_error
        return {"accepted": True}

    def stop_and_hold(self, *, reason: str) -> dict[str, object]:
        self.hold_reasons.append(reason)
        if self.hold_error is not None:
            raise self.hold_error
        return {"held": True}

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _runner_report() -> dict[str, object]:
    return {
        "status": "validated",
        "final_state": TubeState.OBSERVE.value,
        "step_count": 0,
        "hardware_commands_dispatched": False,
    }


def test_live_adapter_loader_closes_nonconforming_factory_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    invalid_adapter = SimpleNamespace(closed=False)

    def close() -> None:
        invalid_adapter.closed = True

    invalid_adapter.close = close
    module = SimpleNamespace(
        make=lambda **_kwargs: invalid_adapter,
    )
    monkeypatch.setattr(cli.importlib, "import_module", lambda _name: module)

    with pytest.raises(cli.ClosedLoopContractError, match="does not implement"):
        cli._load_live_adapter("fixture:make", config={}, output_dir=tmp_path)

    assert invalid_adapter.closed is True


def test_live_adapter_loader_forwards_process_acknowledgements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    captured: dict[str, object] = {}

    class Adapter(cli.ClosedLoopIO):
        @property
        def pose_backend_descriptor(self):
            raise NotImplementedError

        def observe(self, *, state):
            raise NotImplementedError

        def dispatch(self, action):
            raise NotImplementedError

        def stop_and_hold(self, *, reason):
            raise NotImplementedError

        def close(self):
            return None

    adapter = Adapter()

    def make(**kwargs: object) -> object:
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _name: SimpleNamespace(make=make),
    )

    result = cli._load_live_adapter(
        "fixture:make",
        config={"task": "demo06"},
        output_dir=tmp_path,
        accept_arm_only_collision_model=True,
        accept_provisional_camera_extrinsic=True,
    )

    assert result is adapter
    assert captured["accept_arm_only_collision_model"] is True
    assert captured["accept_provisional_camera_extrinsic"] is True


def test_runner_records_close_error_without_hiding_terminal_result(tmp_path: Path) -> None:
    cli = _load_cli()
    controller = _RunnerController(cli, terminal_state=TubeState.COMPLETE)
    io = _RunnerIO(close_error=RuntimeError("camera close failed"))

    report = cli._run_loop(
        controller=controller,
        io=io,
        output_dir=tmp_path,
        report=_runner_report(),
        max_steps=1,
        replay_only=False,
    )

    persisted = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["close_error"] == "camera close failed"
    assert persisted == report
    assert io.closed is True


@pytest.mark.parametrize(
    ("observe_error", "replay_only", "expected_status"),
    [
        (EOFError("stream ended"), False, "observation_stream_ended"),
        (KeyboardInterrupt(), False, "interrupted"),
    ],
)
def test_runner_nonterminal_stream_exit_holds_closes_and_persists(
    tmp_path: Path,
    observe_error: BaseException,
    replay_only: bool,
    expected_status: str,
) -> None:
    cli = _load_cli()
    controller = _RunnerController(cli)
    io = _RunnerIO(observe_error=observe_error)

    report = cli._run_loop(
        controller=controller,
        io=io,
        output_dir=tmp_path,
        report=_runner_report(),
        max_steps=1,
        replay_only=replay_only,
    )

    persisted = json.loads((tmp_path / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == expected_status
    assert report["final_state"] == TubeState.HOLD.value
    assert report["hold_receipt"] == {"held": True}
    assert len(io.hold_reasons) == 1
    assert io.closed is True
    assert persisted == report


def test_runner_step_budget_records_failed_hold_and_still_closes(tmp_path: Path) -> None:
    cli = _load_cli()
    controller = _RunnerController(cli)
    io = _RunnerIO(hold_error=RuntimeError("stop transport failed"))

    report = cli._run_loop(
        controller=controller,
        io=io,
        output_dir=tmp_path,
        report=_runner_report(),
        max_steps=1,
        replay_only=False,
    )

    assert report["status"] == "step_budget_exhausted"
    assert report["hold_receipt"] == {"hold_error": "stop transport failed"}
    assert io.closed is True
    assert json.loads((tmp_path / "run_report.json").read_text()) == report


def test_runner_dispatch_failure_is_logged_before_fail_closed_hold(tmp_path: Path) -> None:
    cli = _load_cli()
    controller = _RunnerController(cli, action_kind=ActionKind.SERVO_TCP)
    io = _RunnerIO(dispatch_error=RuntimeError("servo rejected target"))

    report = cli._run_loop(
        controller=controller,
        io=io,
        output_dir=tmp_path,
        report=_runner_report(),
        max_steps=1,
        replay_only=False,
    )

    step = json.loads((tmp_path / "steps/0000.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed_closed"
    assert report["hardware_commands_dispatched"] is True
    assert report["step_count"] == 1
    assert step["dispatch"] == {
        "status": "failed",
        "receipt": None,
        "error": "servo rejected target",
    }
    assert report["hold_receipt"] == {"held": True}
    assert io.closed is True


def test_replay_mode_logs_unqualified_startup_evidence_atomically(tmp_path: Path) -> None:
    cli = _load_cli()
    config = {
        "schema": CLOSED_LOOP_CONFIG_SCHEMA,
        "task_id": "replay-test",
        "geometry": {
            "relation_id": "relation",
            "T_tube_tcp_demo_grasp": np.eye(4).tolist(),
            "T_socket_tube_entry": np.eye(4).tolist(),
        },
        "pose_backend": {"name": "rgbd_axis_anchor"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    fallback = {
        "name": "rgbd_axis_anchor",
        "available": True,
        "runtime_qualified": False,
        "qualification_id": None,
        "capabilities": [
            "metric_translation",
            "image_axis",
            "source_tube",
        ],
    }
    pose = {
        "object_id": "tube",
        "T_base_object": np.eye(4).tolist(),
        "observed_at_s": 0.99,
        "position_std_m": 0.001,
        "orientation_std_deg": 1.0,
        "backend": fallback,
        "tracking_id": "tube-track",
        "inference_latency_s": 0.01,
        "mask_id": "mask",
        "mask_iou": 0.9,
        "valid_depth_count": 200,
        "foreground_depth_support_count": 120,
        "foreground_depth_support_fraction": 0.6,
        "background_depth_fraction": 0.1,
        "masked_depth_cluster_qualified": True,
        "axis_sign_source": "open_end_detector",
        "cylindrical_roll_canonicalized": True,
    }
    observation = {
        "observation_id": "obs-1",
        "now_s": 1.0,
        "robot": {"T_base_tcp": np.eye(4).tolist(), "motion_complete": True},
        "gripper": {"width_m": 0.04, "moving": False, "fault_free": True},
        "tactile": {"valid": True, "observed_at_s": 0.99},
        "tube_pose": pose,
        "socket_pose": {
            **pose,
            "object_id": "socket",
            "tracking_id": "socket-track",
            "axis_sign_source": "not_applicable",
            "cylindrical_roll_canonicalized": False,
        },
    }
    replay = {
        "schema": CLOSED_LOOP_REPLAY_SCHEMA,
        "pose_backend": fallback,
        "observations": [observation],
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    output = tmp_path / "output"

    args = cli.parse_args(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--replay",
            str(replay_path),
        ]
    )
    assert cli.run(args) == 2
    report = json.loads((output / "run_report.json").read_text(encoding="utf-8"))
    step = json.loads((output / "steps/0000.json").read_text(encoding="utf-8"))
    assert report["status"] == "replay_exhausted"
    assert report["hardware_commands_dispatched"] is False
    assert step["decision"]["next_state"] == "OBSERVE"
    assert step["decision"]["action"]["kind"] == "collect_evidence"
    assert step["dispatch"]["status"] == "not_required"
