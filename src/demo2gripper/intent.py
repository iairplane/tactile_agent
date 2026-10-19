"""Infer a parallel-jaw grasp intent from a tracked hand demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    continuous_symmetry_gauge_rotation,
    invert_transforms,
    longest_true_run,
    normalize,
    project_to_point_cloud,
    robust_direction_mean,
    robust_mean,
    transform_points,
)
from .models import GraspIntent, JsonSerializable, ObservationSequence

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
PINKY_MCP = 17


@dataclass(frozen=True)
class IntentConfig(JsonSerializable):
    max_tip_object_distance_m: float = 0.02
    min_joint_confidence: float = 0.5
    min_pinch_distance_m: float = 0.005
    max_pinch_distance_m: float = 0.12
    min_stable_frames: int = 3
    continuous_symmetry_axis: int | None = None

    def __post_init__(self) -> None:
        if self.continuous_symmetry_axis not in (None, 0, 1, 2):
            raise ValueError("continuous_symmetry_axis must be 0, 1, 2, or None")


def hand_joints_in_object_frame(observation: ObservationSequence) -> np.ndarray:
    """Transform all MANO21 joints from camera coordinates into object coordinates."""

    T_object_camera = invert_transforms(observation.T_camera_object)
    return transform_points(T_object_camera, observation.hand_joints_camera_m)


def stable_contact_mask(
    observation: ObservationSequence,
    config: IntentConfig = IntentConfig(),
) -> np.ndarray:
    joints = hand_joints_in_object_frame(observation)
    tips = joints[:, [THUMB_TIP, INDEX_TIP]]
    distances, _ = project_to_point_cloud(
        tips.reshape(-1, 3), observation.object_points_m
    )
    distances = distances.reshape(-1, 2)
    confidence = observation.hand_joint_confidence[:, [THUMB_TIP, INDEX_TIP]]
    pinch_distance = np.linalg.norm(tips[:, 1] - tips[:, 0], axis=1)
    return (
        np.all(distances <= config.max_tip_object_distance_m, axis=1)
        & np.all(confidence >= config.min_joint_confidence, axis=1)
        & (pinch_distance >= config.min_pinch_distance_m)
        & (pinch_distance <= config.max_pinch_distance_m)
    )


def infer_grasp_intent(
    observation: ObservationSequence,
    config: IntentConfig = IntentConfig(),
) -> GraspIntent:
    """Infer contact geometry and TCP pose from the longest stable pinch."""

    joints = hand_joints_in_object_frame(observation)
    mask = stable_contact_mask(observation, config)
    start, end = longest_true_run(mask)
    if end - start < config.min_stable_frames:
        raise ValueError(
            f"no stable contact interval with at least {config.min_stable_frames} frames"
        )

    tips = joints[start:end, [THUMB_TIP, INDEX_TIP]]
    _, projected = project_to_point_cloud(
        tips.reshape(-1, 3), observation.object_points_m
    )
    projected = projected.reshape(-1, 2, 3)

    frame_centers = np.mean(projected, axis=1)
    closing_directions = projected[:, 1] - projected[:, 0]
    approach_directions = frame_centers - joints[start:end, WRIST]

    if config.continuous_symmetry_axis is not None:
        gauges = np.stack(
            [
                continuous_symmetry_gauge_rotation(
                    config.continuous_symmetry_axis,
                    center,
                    closing,
                    approach,
                )
                for center, closing, approach in zip(
                    frame_centers,
                    closing_directions,
                    approach_directions,
                    strict=True,
                )
            ]
        )
        projected = np.einsum("fij,fkj->fki", gauges, projected)
        frame_centers = np.einsum("fij,fj->fi", gauges, frame_centers)
        closing_directions = np.einsum(
            "fij,fj->fi", gauges, closing_directions
        )
        approach_directions = np.einsum(
            "fij,fj->fi", gauges, approach_directions
        )

    contact_center = robust_mean(frame_centers)
    contact_points = np.stack(
        [robust_mean(projected[:, 0]), robust_mean(projected[:, 1])]
    )

    valid_closing = np.linalg.norm(closing_directions, axis=1) > 1e-9
    if not np.any(valid_closing):
        raise ValueError("projected thumb and index contacts do not define a closing axis")
    x_axis = robust_direction_mean(
        closing_directions[valid_closing], name="closing axis"
    )

    valid_approach = np.linalg.norm(approach_directions, axis=1) > 1e-9
    if not np.any(valid_approach):
        raise ValueError("wrist and contact center do not define an approach axis")
    z_axis = robust_direction_mean(
        approach_directions[valid_approach], name="approach axis"
    )

    x_axis = normalize(x_axis - np.dot(x_axis, z_axis) * z_axis, name="closing axis")
    y_axis = normalize(np.cross(z_axis, x_axis), name="lateral axis")
    x_axis = normalize(np.cross(y_axis, z_axis), name="closing axis")

    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    transform[:3, 3] = contact_center
    widths = np.abs(np.einsum("ij,j->i", closing_directions, x_axis))

    return GraspIntent(
        T_object_gripper_tcp=transform,
        width_m=float(np.median(widths[valid_closing])),
        contact_points_object_m=contact_points,
        stable_interval=(start, end),
        support_frames=end - start,
        continuous_symmetry_axis=config.continuous_symmetry_axis,
    )
