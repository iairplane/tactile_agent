"""Data exchanged between demonstration parsing and grasp selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonSerializable:
    """Mixin for dataclasses whose fields contain NumPy values."""

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True)
class ObservationSequence(JsonSerializable):
    """A hand demonstration and object reconstruction.

    ``T_camera_object`` maps object-frame points into the camera frame. Lengths
    are in metres and timestamps are in seconds.
    """

    T_camera_object: np.ndarray
    hand_joints_camera_m: np.ndarray
    hand_joint_confidence: np.ndarray
    timestamps_s: np.ndarray
    object_points_m: np.ndarray

    def __post_init__(self) -> None:
        frame_count = self.T_camera_object.shape[0]
        expected_shapes = {
            "T_camera_object": (frame_count, 4, 4),
            "hand_joints_camera_m": (frame_count, 21, 3),
            "hand_joint_confidence": (frame_count, 21),
            "timestamps_s": (frame_count,),
        }
        for name, expected in expected_shapes.items():
            actual = np.asarray(getattr(self, name)).shape
            if actual != expected:
                raise ValueError(f"{name} must have shape {expected}, got {actual}")
        if np.asarray(self.object_points_m).ndim != 2 or self.object_points_m.shape[1] != 3:
            raise ValueError("object_points_m must have shape (N, 3)")


@dataclass(frozen=True)
class GraspIntent(JsonSerializable):
    """A parallel-jaw grasp expressed entirely in the object frame.

    The rotation columns of ``T_object_gripper_tcp`` are +X closing, +Y, and
    +Z approach. ``stable_interval`` is half-open: ``[start, end)``.
    """

    T_object_gripper_tcp: np.ndarray
    width_m: float
    contact_points_object_m: np.ndarray
    stable_interval: tuple[int, int]
    support_frames: int
    continuous_symmetry_axis: int | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.T_object_gripper_tcp).shape != (4, 4):
            raise ValueError("T_object_gripper_tcp must have shape (4, 4)")
        if np.asarray(self.contact_points_object_m).shape != (2, 3):
            raise ValueError("contact_points_object_m must have shape (2, 3)")
        if self.continuous_symmetry_axis not in (None, 0, 1, 2):
            raise ValueError("continuous_symmetry_axis must be 0, 1, 2, or None")

    @property
    def contact_center_object_m(self) -> np.ndarray:
        return self.T_object_gripper_tcp[:3, 3]

    @property
    def closing_axis_object(self) -> np.ndarray:
        return self.T_object_gripper_tcp[:3, 0]

    @property
    def approach_axis_object(self) -> np.ndarray:
        return self.T_object_gripper_tcp[:3, 2]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraspIntent":
        return cls(
            T_object_gripper_tcp=np.asarray(data["T_object_gripper_tcp"], dtype=float),
            width_m=float(data["width_m"]),
            contact_points_object_m=np.asarray(
                data["contact_points_object_m"], dtype=float
            ),
            stable_interval=tuple(int(value) for value in data["stable_interval"]),
            support_frames=int(data["support_frames"]),
            continuous_symmetry_axis=(
                None
                if data.get("continuous_symmetry_axis") is None
                else int(data["continuous_symmetry_axis"])
            ),
        )


@dataclass(frozen=True)
class GraspCandidates(JsonSerializable):
    """Candidate TCP poses and pre-grasp jaw openings in the object frame."""

    poses: np.ndarray
    scores: np.ndarray
    widths_m: np.ndarray

    def __post_init__(self) -> None:
        candidate_count = self.poses.shape[0]
        if np.asarray(self.poses).shape != (candidate_count, 4, 4):
            raise ValueError("poses must have shape (K, 4, 4)")
        if np.asarray(self.scores).shape != (candidate_count,):
            raise ValueError("scores must have shape (K,)")
        if np.asarray(self.widths_m).shape != (candidate_count,):
            raise ValueError("widths_m must have shape (K,)")


@dataclass(frozen=True)
class RankedCandidate(JsonSerializable):
    index: int
    T_object_gripper_tcp: np.ndarray
    width_m: float
    model_score: float
    center_error_m: float
    closing_axis_error_rad: float
    approach_axis_error_rad: float
    ranking_cost: float
