#!/usr/bin/env python3
"""Validate an object-relative parallel-jaw grasp in RoboDojo's Isaac Lab.

The command-line and JSON boundary uses XYZW quaternions. Isaac Lab 2.x uses
WXYZ internally. All lengths are metres.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher


def _matrix4(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("T_object_gripper_tcp must be a 4x4 array")
    matrix = [[float(item) for item in row] for row in value]
    if any(len(row) != 4 for row in matrix):
        raise ValueError("T_object_gripper_tcp must be a 4x4 array")
    if any(not math.isfinite(item) for row in matrix for item in row):
        raise ValueError("T_object_gripper_tcp must contain finite values")
    return matrix


def _load_grasp(path: str) -> tuple[list[list[float]], float, float, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    transform = _matrix4(payload["T_object_gripper_tcp"])
    pregrasp_width_m = float(payload["pregrasp_width_m"])
    close_target_width_m = float(payload["close_target_width_m"])
    approach_distance_m = float(payload["approach_distance_m"])
    if not 0.0 <= close_target_width_m <= pregrasp_width_m <= 0.08:
        raise ValueError(
            "gripper widths must satisfy 0 <= close_target_width_m <= "
            "pregrasp_width_m <= 0.08"
        )
    if approach_distance_m < 0.0:
        raise ValueError("approach_distance_m must be non-negative")
    return transform, pregrasp_width_m, close_target_width_m, approach_distance_m


def _normalized_pose(values: list[float] | tuple[float, ...], option: str) -> tuple[float, ...]:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{option} must contain finite values")
    quaternion_norm = math.sqrt(sum(value * value for value in values[3:7]))
    if quaternion_norm == 0.0:
        raise ValueError(f"{option} quaternion must be non-zero")
    return tuple(values[:3]) + tuple(value / quaternion_norm for value in values[3:7])


def _pose_xyzw_to_wxyz(pose: tuple[float, ...]) -> tuple[float, ...]:
    return pose[:3] + (pose[6], pose[3], pose[4], pose[5])


parser = argparse.ArgumentParser(
    description="Validate selected_grasp.json with a Franka Panda in Isaac Lab."
)
parser.add_argument(
    "--grasp-json",
    default="selected_grasp.json",
    help="Selected grasp produced by demo2gripper.",
)
object_group = parser.add_mutually_exclusive_group(required=True)
object_group.add_argument("--object-usd", help="USD whose root prim is one rigid body.")
object_group.add_argument(
    "--smoke-cuboid",
    action="store_true",
    help="Use the built-in 6 x 5 x 8 cm cuboid for an asset-free smoke test.",
)
parser.add_argument(
    "--robot-usd",
    help=(
        "Optional local Franka USD. Use RoboDojo's Assets/Robots/franka/Franka.usd "
        "to avoid fetching the Isaac Lab robot asset."
    ),
)
parser.add_argument(
    "--object-pose",
    nargs=7,
    type=float,
    metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
    default=(0.55, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0),
    help="Ground-truth initial simulated object pose in metres and XYZW order.",
)
target_pose_group = parser.add_mutually_exclusive_group(required=True)
target_pose_group.add_argument(
    "--estimated-object-pose",
    nargs=7,
    type=float,
    metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
    help=(
        "External perception estimate used to build the control target, "
        "in world frame and XYZW order."
    ),
)
target_pose_group.add_argument(
    "--estimated-object-pose-json",
    help="World-frame pose JSON produced by compose_world_object_pose.py.",
)
target_pose_group.add_argument(
    "--use-oracle-object-pose",
    action="store_true",
    help="Debug only: build the control target from the settled simulated object pose.",
)
parser.add_argument(
    "--tcp-offset-m",
    type=float,
    default=0.1034,
    help="Panda hand origin to fingertip-centre TCP distance along hand +Z.",
)
parser.add_argument(
    "--lift-height-m", type=float, default=0.10, help="Commanded world +Z lift distance."
)
parser.add_argument(
    "--lift-success-m", type=float, default=0.05, help="Minimum object displacement for success."
)
parser.add_argument(
    "--contact-threshold-n",
    type=float,
    default=1.0,
    help="Per-finger filtered normal-force threshold.",
)
parser.add_argument(
    "--pin-object-until-contact",
    action="store_true",
    help=(
        "Keep the object at --object-pose through REST, PREGRASP, APPROACH, and "
        "CLOSE, then release it after bilateral finger contact."
    ),
)
parser.add_argument(
    "--result-json", default="isaaclab_grasp_result.json", help="Validation result path."
)
parser.add_argument(
    "--video-output",
    help="Optional 640x480 MP4 showing the grasp phases and final validation result.",
)
parser.add_argument(
    "--rgbd-output-dir",
    help=(
        "Optional directory for one post-REST RGB-D observation, camera calibration, "
        "and evaluation-only object ground truth."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
CAMERA_OUTPUT_ENABLED = (
    args_cli.video_output is not None or args_cli.rgbd_output_dir is not None
)
args_cli.enable_cameras = CAMERA_OUTPUT_ENABLED

try:
    (
        GRASP_TRANSFORM,
        PREGRASP_WIDTH_M,
        CLOSE_TARGET_WIDTH_M,
        APPROACH_DISTANCE_M,
    ) = _load_grasp(args_cli.grasp_json)
except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
    parser.error(f"invalid grasp JSON: {error}")

if args_cli.tcp_offset_m < 0.0:
    parser.error("--tcp-offset-m must be non-negative")
if args_cli.lift_height_m <= 0.0 or args_cli.lift_success_m <= 0.0:
    parser.error("lift distances must be positive")
if args_cli.contact_threshold_n <= 0.0:
    parser.error("contact threshold must be positive")
try:
    OBJECT_SPAWN_POSE_XYZW = _normalized_pose(args_cli.object_pose, "--object-pose")
    if args_cli.estimated_object_pose_json is not None:
        pose_payload = json.loads(
            Path(args_cli.estimated_object_pose_json).read_text(encoding="utf-8")
        )
        pose_values = [float(value) for value in pose_payload["pose_xyzw"]]
        if len(pose_values) != 7:
            raise ValueError("estimated pose JSON pose_xyzw must contain seven values")
        ESTIMATED_OBJECT_POSE_XYZW = _normalized_pose(
            pose_values, "--estimated-object-pose-json"
        )
    elif args_cli.estimated_object_pose is not None:
        ESTIMATED_OBJECT_POSE_XYZW = _normalized_pose(
            args_cli.estimated_object_pose, "--estimated-object-pose"
        )
    else:
        ESTIMATED_OBJECT_POSE_XYZW = None
except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
    parser.error(str(error))

OBJECT_SPAWN_POSE_WXYZ = _pose_xyzw_to_wxyz(OBJECT_SPAWN_POSE_XYZW)
ESTIMATED_OBJECT_POSE_WXYZ = (
    None
    if ESTIMATED_OBJECT_POSE_XYZW is None
    else _pose_xyzw_to_wxyz(ESTIMATED_OBJECT_POSE_XYZW)
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_from_matrix,
    subtract_frame_transforms,
)

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


REST_STEPS = 50
MOVE_STEPS = 300
CLOSE_STEPS = 120
HOLD_STEPS = 150
POSITION_TOLERANCE_M = 0.015
ORIENTATION_TOLERANCE_RAD = math.radians(10.0)
REACHED_STEPS = 5
CONTACT_STABLE_STEPS = 5
SMOKE_CUBOID_SIZE_M = (0.06, 0.05, 0.08)
TABLE_SIZE_M = (1.30, 1.00, 0.05)
TABLE_CENTER = (0.45, 0.0, -0.025)
VIDEO_FPS = 20
VIDEO_CAPTURE_INTERVAL_STEPS = 5
VIDEO_FRAMES: list[tuple[str, Any]] = []
VIDEO_STEP_COUNT = 0


def _object_spawn_cfg():
    if args_cli.smoke_cuboid:
        return sim_utils.CuboidCfg(
            size=SMOKE_CUBOID_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.55, 0.85), metallic=0.1
            ),
            activate_contact_sensors=True,
        )
    return sim_utils.UsdFileCfg(
        usd_path=args_cli.object_usd,
        activate_contact_sensors=True,
    )


def _robot_cfg():
    cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.spawn.activate_contact_sensors = True
    if args_cli.robot_usd is not None:
        cfg.spawn.usd_path = str(Path(args_cli.robot_usd).resolve())
        cfg.spawn.articulation_props.fix_root_link = True
    return cfg


@configclass
class GraspValidationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -TABLE_SIZE_M[2])),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.42, 0.30, 0.20), roughness=0.8
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_CENTER),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = _robot_cfg()

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=_object_spawn_cfg(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBJECT_SPAWN_POSE_WXYZ[:3],
            rot=OBJECT_SPAWN_POSE_WXYZ[3:7],
        ),
    )

    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    if CAMERA_OUTPUT_ENABLED:
        observation_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/ObservationCamera",
            update_period=1.0 / VIDEO_FPS,
            width=640,
            height=480,
            data_types=(
                ["rgb", "distance_to_image_plane"]
                if args_cli.rgbd_output_dir is not None
                else ["rgb"]
            ),
            depth_clipping_behavior="zero",
            update_latest_camera_pose=args_cli.rgbd_output_dir is not None,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
        )


def _hand_pose_from_canonical_tcp(
    tcp_pos_w: torch.Tensor, tcp_quat_w: torch.Tensor
) -> torch.Tensor:
    """Map canonical +X-closing TCP pose to the Panda hand body pose."""
    zero = torch.zeros((1, 3), device=tcp_pos_w.device)
    half_angle = -0.25 * math.pi
    tcp_to_panda_quat = torch.tensor(
        [[math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]],
        device=tcp_pos_w.device,
    )
    panda_tcp_pos, panda_tcp_quat = combine_frame_transforms(
        tcp_pos_w, tcp_quat_w, zero, tcp_to_panda_quat
    )
    tcp_to_hand = torch.tensor(
        [[0.0, 0.0, -args_cli.tcp_offset_m]], device=tcp_pos_w.device
    )
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=tcp_pos_w.device)
    hand_pos, hand_quat = combine_frame_transforms(
        panda_tcp_pos, panda_tcp_quat, tcp_to_hand, identity
    )
    return torch.cat((hand_pos, hand_quat), dim=-1)


def _grasp_targets(object_pose_w: torch.Tensor) -> dict[str, torch.Tensor]:
    device = object_pose_w.device
    transform = torch.tensor(GRASP_TRANSFORM, dtype=torch.float32, device=device)
    relative_pos = transform[:3, 3].unsqueeze(0)
    relative_quat = quat_from_matrix(transform[:3, :3].unsqueeze(0))
    grasp_pos, grasp_quat = combine_frame_transforms(
        object_pose_w[:, :3], object_pose_w[:, 3:7], relative_pos, relative_quat
    )

    retreat = torch.tensor(
        [[0.0, 0.0, -APPROACH_DISTANCE_M]], dtype=torch.float32, device=device
    )
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    pregrasp_pos, pregrasp_quat = combine_frame_transforms(
        grasp_pos, grasp_quat, retreat, identity
    )
    lift_pos = grasp_pos + torch.tensor(
        [[0.0, 0.0, args_cli.lift_height_m]], device=device
    )
    return {
        "pregrasp": _hand_pose_from_canonical_tcp(pregrasp_pos, pregrasp_quat),
        "grasp": _hand_pose_from_canonical_tcp(grasp_pos, grasp_quat),
        "lift": _hand_pose_from_canonical_tcp(lift_pos, grasp_quat),
    }


def _set_ik_target(
    controller: DifferentialIKController,
    robot: Any,
    target_hand_w: torch.Tensor,
) -> None:
    root_pose_w = robot.data.root_pose_w
    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        target_hand_w[:, :3],
        target_hand_w[:, 3:7],
    )
    controller.reset()
    controller.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))


def _pin_object_to_spawn(scene: InteractiveScene) -> None:
    object_asset = scene["object"]
    pose = torch.tensor(
        [OBJECT_SPAWN_POSE_WXYZ], dtype=torch.float32, device=object_asset.device
    )
    object_asset.write_root_pose_to_sim(pose)
    object_asset.write_root_velocity_to_sim(
        torch.zeros((scene.num_envs, 6), device=object_asset.device)
    )


def _capture_video_frame(scene: InteractiveScene, phase: str) -> None:
    if args_cli.video_output is None:
        return
    global VIDEO_STEP_COUNT
    VIDEO_STEP_COUNT += 1
    if VIDEO_STEP_COUNT % VIDEO_CAPTURE_INTERVAL_STEPS != 0:
        return
    rgb = scene["observation_camera"].data.output.get("rgb")
    if rgb is None:
        return
    VIDEO_FRAMES.append((phase, rgb[0, ..., :3].detach().cpu().numpy().copy()))


def _write_video(result: dict[str, Any]) -> None:
    if args_cli.video_output is None or not VIDEO_FRAMES:
        return
    import cv2

    output = Path(args_cli.video_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = VIDEO_FRAMES[0][1].shape[:2]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    outcomes = (
        ("IK", result["ik_reachable"]),
        ("2-FINGER CONTACT", result["bilateral_contact"]),
        ("LIFT", result["lifted"]),
        ("HOLD", result["held"]),
    )
    for phase, rgb in VIDEO_FRAMES:
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(frame, (0, 0), (width, 104), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"PHASE: {phase}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for index, (label, passed) in enumerate(outcomes):
            x = 18 + (index % 2) * 310
            y = 61 + (index // 2) * 31
            color = (60, 210, 60) if passed else (60, 60, 230)
            cv2.putText(
                frame,
                f"{label}: {'PASS' if passed else 'FAIL'}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
        writer.write(frame)
    writer.release()
    print(f"[INFO] video={output}")


def _write_rgbd_snapshot(
    scene: InteractiveScene, ground_truth_object_pose: torch.Tensor
) -> None:
    if args_cli.rgbd_output_dir is None:
        return
    import cv2

    camera = scene["observation_camera"]
    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    depth = (
        camera.data.output["distance_to_image_plane"][0, ..., 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    intrinsics = (
        camera.data.intrinsic_matrices[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    world_camera = torch.eye(4, dtype=torch.float32, device=scene.device)
    world_camera[:3, :3] = matrix_from_quat(camera.data.quat_w_ros[0])
    world_camera[:3, 3] = camera.data.pos_w[0]

    world_object = torch.eye(4, dtype=torch.float32, device=scene.device)
    world_object[:3, :3] = matrix_from_quat(ground_truth_object_pose[0, 3:7])
    world_object[:3, 3] = ground_truth_object_pose[0, :3]

    output = Path(args_cli.rgbd_output_dir)
    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    (output / "calibration").mkdir(parents=True, exist_ok=True)
    (output / "evaluation").mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output / "rgb" / "000000.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError("could not write RGB snapshot")
    np.save(output / "depth" / "000000.npy", depth, allow_pickle=False)
    np.save(output / "K.npy", intrinsics, allow_pickle=False)
    np.save(
        output / "calibration" / "T_world_camera.npy",
        world_camera.detach().cpu().numpy(),
        allow_pickle=False,
    )
    np.save(
        output / "evaluation" / "T_world_object.npy",
        world_object.detach().cpu().numpy(),
        allow_pickle=False,
    )
    print(f"[INFO] rgbd_snapshot={output}")


def _step_ik(
    phase: str,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    controller: DifferentialIKController,
    arm: SceneEntityCfg,
    finger: SceneEntityCfg,
    ee_jacobian_index: int,
    finger_position_m: float,
) -> None:
    robot = scene["robot"]
    jacobian = robot.root_physx_view.get_jacobians()[
        :, ee_jacobian_index, :, arm.joint_ids
    ]
    hand_pose_w = robot.data.body_pose_w[:, arm.body_ids[0]]
    root_pose_w = robot.data.root_pose_w
    arm_joint_pos = robot.data.joint_pos[:, arm.joint_ids]
    hand_pos_b, hand_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3],
        root_pose_w[:, 3:7],
        hand_pose_w[:, :3],
        hand_pose_w[:, 3:7],
    )
    arm_target = controller.compute(hand_pos_b, hand_quat_b, jacobian, arm_joint_pos)
    finger_target = torch.full(
        (scene.num_envs, len(finger.joint_ids)),
        finger_position_m,
        dtype=torch.float32,
        device=robot.device,
    )
    robot.set_joint_position_target(arm_target, joint_ids=arm.joint_ids)
    robot.set_joint_position_target(finger_target, joint_ids=finger.joint_ids)
    if args_cli.pin_object_until_contact and phase in {
        "PREGRASP",
        "APPROACH",
        "CLOSE",
    }:
        _pin_object_to_spawn(scene)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    _capture_video_frame(scene, phase)


def _step_rest(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    arm: SceneEntityCfg,
    finger: SceneEntityCfg,
) -> None:
    robot = scene["robot"]
    arm_target = robot.data.default_joint_pos[:, arm.joint_ids]
    finger_target = torch.full(
        (scene.num_envs, len(finger.joint_ids)),
        0.5 * PREGRASP_WIDTH_M,
        device=robot.device,
    )
    robot.set_joint_position_target(arm_target, joint_ids=arm.joint_ids)
    robot.set_joint_position_target(finger_target, joint_ids=finger.joint_ids)
    if args_cli.pin_object_until_contact:
        _pin_object_to_spawn(scene)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    _capture_video_frame(scene, "REST")


def _pose_reached(scene: InteractiveScene, arm: SceneEntityCfg, target_w: torch.Tensor) -> bool:
    current = scene["robot"].data.body_pose_w[:, arm.body_ids[0]]
    position_error = torch.linalg.vector_norm(current[:, :3] - target_w[:, :3], dim=-1)
    quaternion_dot = torch.sum(current[:, 3:7] * target_w[:, 3:7], dim=-1).abs()
    orientation_error = 2.0 * torch.acos(torch.clamp(quaternion_dot, 0.0, 1.0))
    return bool(
        torch.all(position_error <= POSITION_TOLERANCE_M)
        and torch.all(orientation_error <= ORIENTATION_TOLERANCE_RAD)
    )


def _filtered_force_n(scene: InteractiveScene, sensor_name: str) -> float:
    force_matrix = scene[sensor_name].data.force_matrix_w
    if force_matrix is None:
        return 0.0
    return float(torch.linalg.vector_norm(force_matrix, dim=-1).amax().item())


def _update_metrics(
    scene: InteractiveScene, metrics: dict[str, Any], initial_object_z: float
) -> tuple[float, float]:
    left_force = _filtered_force_n(scene, "left_finger_contact")
    right_force = _filtered_force_n(scene, "right_finger_contact")
    metrics["max_left_force_n"] = max(metrics["max_left_force_n"], left_force)
    metrics["max_right_force_n"] = max(metrics["max_right_force_n"], right_force)
    metrics["bilateral_contact"] |= (
        left_force >= args_cli.contact_threshold_n
        and right_force >= args_cli.contact_threshold_n
    )
    object_lift = float(scene["object"].data.root_pos_w[0, 2].item()) - initial_object_z
    metrics["max_object_lift_m"] = max(metrics["max_object_lift_m"], object_lift)
    return left_force, right_force


def _move_to(
    phase: str,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    controller: DifferentialIKController,
    arm: SceneEntityCfg,
    finger: SceneEntityCfg,
    ee_jacobian_index: int,
    target_w: torch.Tensor,
    finger_position_m: float,
    metrics: dict[str, Any] | None = None,
    initial_object_z: float = 0.0,
) -> tuple[bool, int]:
    print(f"[INFO] phase={phase}")
    _set_ik_target(controller, scene["robot"], target_w)
    consecutive = 0
    for step in range(1, MOVE_STEPS + 1):
        if not simulation_app.is_running():
            return False, step - 1
        _step_ik(
            phase,
            sim,
            scene,
            controller,
            arm,
            finger,
            ee_jacobian_index,
            finger_position_m,
        )
        if metrics is not None:
            _update_metrics(scene, metrics, initial_object_z)
        consecutive = consecutive + 1 if _pose_reached(scene, arm, target_w) else 0
        if consecutive >= REACHED_STEPS:
            return True, step
    return False, MOVE_STEPS


def _hold_phase(
    phase: str,
    steps: int,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    controller: DifferentialIKController,
    arm: SceneEntityCfg,
    finger: SceneEntityCfg,
    ee_jacobian_index: int,
    target_w: torch.Tensor,
    finger_position_m: float,
    metrics: dict[str, Any],
    initial_object_z: float,
) -> int:
    print(f"[INFO] phase={phase}")
    _set_ik_target(controller, scene["robot"], target_w)
    completed = 0
    for _ in range(steps):
        if not simulation_app.is_running():
            break
        _step_ik(
            phase,
            sim,
            scene,
            controller,
            arm,
            finger,
            ee_jacobian_index,
            finger_position_m,
        )
        _update_metrics(scene, metrics, initial_object_z)
        completed += 1
    return completed


def _close_until_bilateral_contact(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    controller: DifferentialIKController,
    arm: SceneEntityCfg,
    finger: SceneEntityCfg,
    ee_jacobian_index: int,
    target_w: torch.Tensor,
    metrics: dict[str, Any],
    initial_object_z: float,
) -> tuple[bool, int, float]:
    print("[INFO] phase=CLOSE")
    _set_ik_target(controller, scene["robot"], target_w)
    commanded_width_m = PREGRASP_WIDTH_M
    completed = 0
    stable_contact_steps = 0
    for step in range(1, CLOSE_STEPS + 1):
        if not simulation_app.is_running():
            break
        fraction = step / CLOSE_STEPS
        commanded_width_m = PREGRASP_WIDTH_M + fraction * (
            CLOSE_TARGET_WIDTH_M - PREGRASP_WIDTH_M
        )
        _step_ik(
            "CLOSE",
            sim,
            scene,
            controller,
            arm,
            finger,
            ee_jacobian_index,
            0.5 * commanded_width_m,
        )
        left_force, right_force = _update_metrics(scene, metrics, initial_object_z)
        completed = step
        if (
            left_force >= args_cli.contact_threshold_n
            and right_force >= args_cli.contact_threshold_n
        ):
            stable_contact_steps += 1
        else:
            stable_contact_steps = 0
        if stable_contact_steps >= CONTACT_STABLE_STEPS:
            return True, completed, commanded_width_m
    return False, completed, commanded_width_m


def _write_result(result: dict[str, Any]) -> None:
    output = Path(args_cli.result_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[INFO] result={output}")


def _pose_tensor_wxyz_to_xyzw(pose: torch.Tensor) -> list[float]:
    values = pose.tolist()
    return values[:3] + values[4:7] + [values[3]]


def run_validation(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    arm = SceneEntityCfg(
        "robot", joint_names=["panda_joint.*"], body_names=["panda_hand"]
    )
    finger = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    arm.resolve(scene)
    finger.resolve(scene)
    ee_jacobian_index = arm.body_ids[0] - 1 if robot.is_fixed_base else arm.body_ids[0]
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        num_envs=scene.num_envs,
        device=sim.device,
    )

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    object_pose = torch.tensor(
        [OBJECT_SPAWN_POSE_WXYZ], dtype=torch.float32, device=robot.device
    )
    scene["object"].write_root_pose_to_sim(object_pose)
    scene["object"].write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
    scene.reset()

    print("[INFO] phase=REST")
    rest_completed = 0
    for _ in range(REST_STEPS):
        if not simulation_app.is_running():
            break
        _step_rest(sim, scene, arm, finger)
        rest_completed += 1

    ground_truth_object_pose = scene["object"].data.root_pose_w.clone()
    _write_rgbd_snapshot(scene, ground_truth_object_pose)
    if ESTIMATED_OBJECT_POSE_WXYZ is not None:
        target_object_pose = torch.tensor(
            [ESTIMATED_OBJECT_POSE_WXYZ], dtype=torch.float32, device=robot.device
        )
        target_pose_source = "external_estimate"
    elif args_cli.pin_object_until_contact:
        target_object_pose = object_pose.clone()
        target_pose_source = "spawn_pose_pinned_until_contact"
    else:
        target_object_pose = ground_truth_object_pose.clone()
        target_pose_source = "sim_oracle_after_rest"
    print(f"[INFO] target_pose_source={target_pose_source}")
    targets = _grasp_targets(target_object_pose)
    initial_object_z = float(ground_truth_object_pose[0, 2].item())
    metrics: dict[str, Any] = {
        "bilateral_contact": False,
        "max_left_force_n": 0.0,
        "max_right_force_n": 0.0,
        "max_object_lift_m": 0.0,
    }
    phase_steps: dict[str, int] = {"REST": rest_completed}

    pregrasp_reached, phase_steps["PREGRASP"] = _move_to(
        "PREGRASP",
        sim,
        scene,
        controller,
        arm,
        finger,
        ee_jacobian_index,
        targets["pregrasp"],
        0.5 * PREGRASP_WIDTH_M,
    )
    approach_reached = False
    close_contact_reached = False
    grasp_command_width_m = PREGRASP_WIDTH_M
    lift_target_reached = False
    final_phase = "PREGRASP"
    if pregrasp_reached:
        approach_reached, phase_steps["APPROACH"] = _move_to(
            "APPROACH",
            sim,
            scene,
            controller,
            arm,
            finger,
            ee_jacobian_index,
            targets["grasp"],
            0.5 * PREGRASP_WIDTH_M,
        )
        final_phase = "APPROACH"
    if pregrasp_reached and approach_reached:
        (
            close_contact_reached,
            phase_steps["CLOSE"],
            grasp_command_width_m,
        ) = _close_until_bilateral_contact(
            sim,
            scene,
            controller,
            arm,
            finger,
            ee_jacobian_index,
            targets["grasp"],
            metrics,
            initial_object_z,
        )
        final_phase = "CLOSE"
    if close_contact_reached:
        lift_target_reached, phase_steps["LIFT"] = _move_to(
            "LIFT",
            sim,
            scene,
            controller,
            arm,
            finger,
            ee_jacobian_index,
            targets["lift"],
            0.5 * grasp_command_width_m,
            metrics,
            initial_object_z,
        )
        final_phase = "LIFT"
        phase_steps["HOLD"] = _hold_phase(
            "HOLD",
            HOLD_STEPS,
            sim,
            scene,
            controller,
            arm,
            finger,
            ee_jacobian_index,
            targets["lift"],
            0.5 * grasp_command_width_m,
            metrics,
            initial_object_z,
        )
        final_phase = "HOLD"

    final_left_force = _filtered_force_n(scene, "left_finger_contact")
    final_right_force = _filtered_force_n(scene, "right_finger_contact")
    final_object_lift_m = (
        float(scene["object"].data.root_pos_w[0, 2].item()) - initial_object_z
    )
    ik_reachable = pregrasp_reached and approach_reached
    lifted = metrics["max_object_lift_m"] >= args_cli.lift_success_m
    held = (
        final_object_lift_m >= args_cli.lift_success_m
        and final_left_force >= args_cli.contact_threshold_n
        and final_right_force >= args_cli.contact_threshold_n
    )
    result = {
        "schema_version": 1,
        "ik_reachable": bool(ik_reachable),
        "bilateral_contact": bool(metrics["bilateral_contact"]),
        "lifted": bool(lifted),
        "held": bool(held),
        "final_phase": final_phase,
        "close_contact_reached": bool(close_contact_reached),
        "lift_target_reached": bool(lift_target_reached),
        "phase_steps": phase_steps,
        "object_lift_m": final_object_lift_m,
        "max_object_lift_m": metrics["max_object_lift_m"],
        "final_contact_force_n": {
            "left": final_left_force,
            "right": final_right_force,
        },
        "max_contact_force_n": {
            "left": metrics["max_left_force_n"],
            "right": metrics["max_right_force_n"],
        },
        "target_pose_source": target_pose_source,
        "target_object_pose_xyzw": _pose_tensor_wxyz_to_xyzw(target_object_pose[0]),
        "spawn_object_pose_xyzw": list(OBJECT_SPAWN_POSE_XYZW),
        "ground_truth_object_pose_after_rest_xyzw": _pose_tensor_wxyz_to_xyzw(
            ground_truth_object_pose[0]
        ),
        "pregrasp_width_m": PREGRASP_WIDTH_M,
        "close_target_width_m": CLOSE_TARGET_WIDTH_M,
        "contact_command_width_m": grasp_command_width_m,
        "contact_threshold_n": args_cli.contact_threshold_n,
        "contact_stable_steps": CONTACT_STABLE_STEPS,
        "approach_distance_m": APPROACH_DISTANCE_M,
        "tcp_offset_m": args_cli.tcp_offset_m,
        "canonical_tcp_to_panda_hand_yaw_deg": -90.0,
        "pin_object_until_contact": bool(args_cli.pin_object_until_contact),
        "object_release_phase": (
            "BILATERAL_CONTACT" if args_cli.pin_object_until_contact else None
        ),
    }
    if args_cli.video_output is not None:
        result["video_output"] = str(Path(args_cli.video_output).resolve())
    _write_result(result)
    _write_video(result)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,
        device=args_cli.device,
        render_interval=(
            VIDEO_CAPTURE_INTERVAL_STEPS if args_cli.video_output is not None else 1
        ),
        save_logs_to_file=False,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.6, 1.2, 1.1), target=(0.45, 0.0, 0.25))
    scene = InteractiveScene(GraspValidationSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    if CAMERA_OUTPUT_ENABLED:
        camera_position = torch.tensor(
            [[1.25, 0.90, 1.05]], dtype=torch.float32, device=sim.device
        )
        camera_target = torch.tensor(
            [[0.50, 0.0, 0.16]], dtype=torch.float32, device=sim.device
        )
        scene["observation_camera"].set_world_poses_from_view(
            camera_position, camera_target
        )
    print("[INFO] Isaac Lab scene ready")
    print("[INFO] table_top_z=0.0")
    print("[INFO] canonical TCP +X closing -> Panda hand +Y via Rz(-90 deg)")
    run_validation(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close(skip_cleanup=True)
