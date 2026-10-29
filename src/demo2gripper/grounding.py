"""Ground robot-independent stage trajectories in the current scene.

The functions in this module operate only on NumPy pose trajectories.  They do
not decide which stage should run and do not emit robot commands.  A trajectory
is an array of ``T_parent_tcp`` poses shaped ``(N, 4, 4)``.

Translation-only and gravity grounding deliberately leave waypoint rotations
unchanged.  Their results therefore report ``rotation_relocalized=False`` and
cannot be consumed through :meth:`GroundedTrajectory.require_full_pose`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

GroundingMethod = Literal["translation_only", "se3", "gravity"]


@dataclass(frozen=True)
class GroundedPositionPath:
    """A grounded ``(N, 3)`` path with deliberately unresolved orientation.

    This type cannot be passed off as a pose trajectory: it exposes
    ``positions_base_m`` rather than ``poses`` and always rejects
    :meth:`require_full_pose`.
    """

    positions_base_m: np.ndarray
    anchor_kind: str
    representation: str = field(default="position_offsets_base_m", init=False)
    position_relocalized: bool = field(default=True, init=False)
    orientation_resolved: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        positions = _validate_position_path(
            self.positions_base_m, name="positions_base_m"
        )
        object.__setattr__(self, "positions_base_m", positions)
        object.__setattr__(self, "anchor_kind", _validate_anchor_kind(self.anchor_kind))

    def require_full_pose(self) -> np.ndarray:
        raise ValueError(
            "position-offset path has unresolved orientation and no full-pose trajectory"
        )


@dataclass(frozen=True)
class GroundedTrajectory:
    """A grounded pose trajectory with explicit relocation guarantees."""

    poses: np.ndarray
    method: GroundingMethod
    anchor_kind: str
    position_relocalized: bool
    rotation_relocalized: bool

    def require_full_pose(self) -> np.ndarray:
        """Return a copy only when both position and rotation were relocated."""

        if not self.position_relocalized or not self.rotation_relocalized:
            raise ValueError(
                "trajectory is not fully pose-relocated: "
                f"method={self.method}, position={self.position_relocalized}, "
                f"rotation={self.rotation_relocalized}"
            )
        return self.poses.copy()


def _validate_rigid_transform(transform: np.ndarray, *, name: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform


def _validate_trajectory(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (N, 4, 4), got {poses.shape}")
    if len(poses) == 0:
        raise ValueError("poses must contain at least one waypoint")
    for index, pose in enumerate(poses):
        _validate_rigid_transform(pose, name=f"poses[{index}]")
    return poses


def _validate_position(position_m: np.ndarray, *, name: str) -> np.ndarray:
    position_m = np.asarray(position_m, dtype=float)
    if position_m.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {position_m.shape}")
    if not np.isfinite(position_m).all():
        raise ValueError(f"{name} contains non-finite values")
    return position_m


def _validate_position_path(positions_m: np.ndarray, *, name: str) -> np.ndarray:
    positions_m = np.asarray(positions_m, dtype=float)
    if positions_m.ndim != 2 or positions_m.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3), got {positions_m.shape}")
    if len(positions_m) == 0:
        raise ValueError(f"{name} must contain at least one waypoint")
    if not np.isfinite(positions_m).all():
        raise ValueError(f"{name} contains non-finite values")
    return positions_m


def _validate_anchor_kind(anchor_kind: str) -> str:
    if not isinstance(anchor_kind, str) or not anchor_kind.strip():
        raise ValueError("anchor_kind must be a non-empty string")
    return anchor_kind


def _resolve_waypoint_index(length: int, waypoint_index: int) -> int:
    index = waypoint_index if waypoint_index >= 0 else length + waypoint_index
    if not 0 <= index < length:
        raise ValueError(
            f"waypoint_index {waypoint_index} is outside a trajectory of length {length}"
        )
    return index


def ground_position_offsets_base(
    offsets_base_m: np.ndarray,
    *,
    live_anchor_position_m: np.ndarray,
    anchor_kind: str = "object_center_base_aligned",
) -> GroundedPositionPath:
    """Add base-aligned ``(N, 3)`` offsets to a live position anchor.

    No orientation is present in the input or output.  In particular, callers
    must not insert identity rotations to convert this result into poses.
    """

    offsets = _validate_position_path(offsets_base_m, name="offsets_base_m")
    live_anchor = _validate_position(
        live_anchor_position_m, name="live_anchor_position_m"
    )
    return GroundedPositionPath(
        positions_base_m=live_anchor[None, :] + offsets,
        anchor_kind=anchor_kind,
    )


def ground_trajectory_by_translation(
    poses: np.ndarray,
    *,
    demo_anchor_position_m: np.ndarray,
    live_anchor_position_m: np.ndarray,
    anchor_kind: str = "object",
) -> GroundedTrajectory:
    """Move waypoint positions by a live-anchor translation.

    Rotations are copied byte-for-byte from the demonstration.  This mode is
    suitable when perception provides only a live object centre or another 3-D
    anchor.  It does not establish that the live object orientation matches the
    demonstration.
    """

    trajectory = _validate_trajectory(poses)
    demo_anchor = _validate_position(demo_anchor_position_m, name="demo_anchor_position_m")
    live_anchor = _validate_position(live_anchor_position_m, name="live_anchor_position_m")
    anchor_kind = _validate_anchor_kind(anchor_kind)

    grounded = trajectory.copy()
    grounded[:, :3, 3] += live_anchor - demo_anchor
    return GroundedTrajectory(
        poses=grounded,
        method="translation_only",
        anchor_kind=anchor_kind,
        position_relocalized=True,
        rotation_relocalized=False,
    )


def ground_trajectory_by_se3(
    poses: np.ndarray,
    *,
    T_demo_anchor: np.ndarray,
    T_live_anchor: np.ndarray,
    anchor_kind: str = "object",
) -> GroundedTrajectory:
    """Rigidly relocate every waypoint from a demo anchor to a live anchor."""

    trajectory = _validate_trajectory(poses)
    demo_anchor = _validate_rigid_transform(T_demo_anchor, name="T_demo_anchor")
    live_anchor = _validate_rigid_transform(T_live_anchor, name="T_live_anchor")
    anchor_kind = _validate_anchor_kind(anchor_kind)

    T_live_demo = live_anchor @ np.linalg.inv(demo_anchor)
    grounded = T_live_demo[None, ...] @ trajectory
    for index, pose in enumerate(grounded):
        _validate_rigid_transform(pose, name=f"grounded_poses[{index}]")
    return GroundedTrajectory(
        poses=grounded,
        method="se3",
        anchor_kind=anchor_kind,
        position_relocalized=True,
        rotation_relocalized=True,
    )


def _shortest_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm <= 1e-9 or target_norm <= 1e-9:
        raise ValueError("gravity vectors must be non-zero")
    source = source / source_norm
    target = target / target_norm
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-10:
        return np.eye(3)
    if cosine < -1.0 + 1e-10:
        raise ValueError("opposite gravity directions do not define a unique alignment axis")
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / sine**2)


def ground_trajectory_by_gravity(
    poses: np.ndarray,
    *,
    demo_anchor_position_m: np.ndarray,
    live_anchor_position_m: np.ndarray,
    demo_gravity: np.ndarray,
    live_gravity: np.ndarray,
    anchor_kind: str = "object",
) -> GroundedTrajectory:
    """Align waypoint positions with live gravity around a live anchor.

    Gravity fixes pitch and roll of the path geometry but leaves yaw
    unobservable.  Consequently this function transforms positions only and
    retains the demonstrated waypoint rotations with
    ``rotation_relocalized=False``.
    """

    trajectory = _validate_trajectory(poses)
    demo_anchor = _validate_position(demo_anchor_position_m, name="demo_anchor_position_m")
    live_anchor = _validate_position(live_anchor_position_m, name="live_anchor_position_m")
    demo_gravity = _validate_position(demo_gravity, name="demo_gravity")
    live_gravity = _validate_position(live_gravity, name="live_gravity")
    anchor_kind = _validate_anchor_kind(anchor_kind)

    alignment = _shortest_rotation(demo_gravity, live_gravity)
    offsets = trajectory[:, :3, 3] - demo_anchor
    grounded = trajectory.copy()
    grounded[:, :3, 3] = live_anchor + np.einsum("ij,nj->ni", alignment, offsets)
    return GroundedTrajectory(
        poses=grounded,
        method="gravity",
        anchor_kind=anchor_kind,
        position_relocalized=True,
        rotation_relocalized=False,
    )


def ground_trajectory_to_position_anchor(
    poses: np.ndarray,
    *,
    live_anchor_position_m: np.ndarray,
    waypoint_index: int = -1,
    anchor_kind: Literal["release", "placement"] = "placement",
) -> GroundedTrajectory:
    """Translate a release or placement waypoint onto a live 3-D anchor."""

    if anchor_kind not in ("release", "placement"):
        raise ValueError("anchor_kind must be 'release' or 'placement'")
    trajectory = _validate_trajectory(poses)
    index = _resolve_waypoint_index(len(trajectory), waypoint_index)
    return ground_trajectory_by_translation(
        trajectory,
        demo_anchor_position_m=trajectory[index, :3, 3],
        live_anchor_position_m=live_anchor_position_m,
        anchor_kind=anchor_kind,
    )


def ground_trajectory_to_pose_anchor(
    poses: np.ndarray,
    *,
    T_live_anchor: np.ndarray,
    waypoint_index: int = -1,
    anchor_kind: Literal["release", "placement"] = "placement",
) -> GroundedTrajectory:
    """Rigidly place a release or placement waypoint at a live pose anchor."""

    if anchor_kind not in ("release", "placement"):
        raise ValueError("anchor_kind must be 'release' or 'placement'")
    trajectory = _validate_trajectory(poses)
    index = _resolve_waypoint_index(len(trajectory), waypoint_index)
    return ground_trajectory_by_se3(
        trajectory,
        T_demo_anchor=trajectory[index],
        T_live_anchor=T_live_anchor,
        anchor_kind=anchor_kind,
    )
