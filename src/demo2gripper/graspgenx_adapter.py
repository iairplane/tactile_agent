"""Pure geometry and artifact helpers for GraspGenX Hand-E candidates.

GraspGenX predicts poses of its *canonical grasp frame*: local ``+X`` is the
jaw-closing axis and local ``+Z`` is the approach axis.  That frame is not the
native Robotiq Hand-E ``base_link`` frame (native ``+Y`` is the jaw axis and
native ``+X`` points toward the fingers).  The Hand-E assets are already
rotated into the GraspGenX convention, so applying the native-base rotation to
sampler output would rotate every candidate a second time.

This project ranks finger-centre TCP poses.  Each canonical prediction is
therefore restored from the centred inference frame to the input object frame
and translated to the fingertip centre along canonical local ``+Z``.  Its
orientation is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FRAME_CONVENTION = "T_object_gripper_tcp; +X closing, +Z approach"
GRIPPER_NAME = "robotiq_hande"

# Hand-E's GraspGenX URDF fixes its native base_link below the canonical
# ``world`` frame with rpy=(0, -pi/2, +pi/2):
#
#     p_canonical = R_CANONICAL_HANDE_BASE @ p_hande_base
#
# Columns show native +X -> canonical +Z, native +Y -> canonical -X and
# native +Z -> canonical -Y.  Consequently canonical +X is native -Y (the
# unsigned jaw-closing axis) and canonical +Z is native +X (approach).
R_CANONICAL_HANDE_BASE = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
R_HANDE_BASE_CANONICAL_TCP = R_CANONICAL_HANDE_BASE.T


def hande_base_to_canonical_tcp(tcp_offset_m: float) -> np.ndarray:
    """Return ``T_hande_base_canonical_tcp`` from the shipped Hand-E asset.

    This transform documents the native Hand-E frame mapping.  It is useful
    when connecting the asset to a robot model, but it must **not** be applied
    to GraspGenX sampler poses: sampler poses are canonical already.
    """

    if not np.isfinite(tcp_offset_m) or tcp_offset_m <= 0.0:
        raise ValueError("tcp_offset_m must be finite and positive")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R_HANDE_BASE_CANONICAL_TCP
    # Canonical TCP +Z is native base +X.
    transform[:3, 3] = [float(tcp_offset_m), 0.0, 0.0]
    return transform


def validate_object_points(points: np.ndarray) -> np.ndarray:
    """Return a finite ``(N, 3)`` float array suitable for inference."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"object points must have shape (N, 3), got {points.shape}")
    if len(points) < 3:
        raise ValueError("object points must contain at least 3 points")
    if not np.all(np.isfinite(points)):
        raise ValueError("object points contain non-finite values")
    return points


def center_object_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centre object-frame points and return ``(points_centered, T_object_centered)``.

    The centred frame keeps the object's axes and moves its origin to the point
    cloud centroid, so ``T_object_centered`` is a pure translation by that
    centroid.  It is the exact inverse transform needed after inference.
    """

    points = validate_object_points(points)
    center = np.mean(points, axis=0)
    centered = points - center
    T_object_centered = np.eye(4, dtype=np.float64)
    T_object_centered[:3, 3] = center
    return centered.astype(np.float32), T_object_centered


def canonical_grasp_poses_to_object_tcp(
    poses_centered_canonical: np.ndarray,
    T_object_centered: np.ndarray,
    tcp_offset_m: float,
) -> np.ndarray:
    """Restore centred GraspGenX poses and translate to the finger TCP.

    ``poses_centered_canonical`` contains ``T_centered_graspgenx_canonical``;
    its local ``+X``/``+Z`` are already closing/approach.  The result is
    ``T_object_gripper_tcp``.  The TCP offset is applied along canonical local
    ``+Z`` by right-multiplication, with no additional rotation.
    """

    poses = np.asarray(poses_centered_canonical, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (K, 4, 4), got {poses.shape}")
    T_object_centered = np.asarray(T_object_centered, dtype=np.float64)
    if T_object_centered.shape != (4, 4):
        raise ValueError("T_object_centered must have shape (4, 4)")
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(T_object_centered)):
        raise ValueError("poses and T_object_centered must be finite")
    if not np.isfinite(tcp_offset_m) or tcp_offset_m <= 0.0:
        raise ValueError("tcp_offset_m must be finite and positive")
    if len(poses) == 0:
        raise ValueError("GraspGenX returned no candidate poses")
    expected_last_row = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (len(poses), 4))
    if not np.allclose(poses[:, 3, :], expected_last_row, atol=1e-5):
        raise ValueError("candidate poses are not homogeneous transforms")
    if not np.allclose(T_object_centered[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T_object_centered is not a homogeneous transform")

    T_canonical_tcp = np.eye(4, dtype=np.float64)
    T_canonical_tcp[2, 3] = float(tcp_offset_m)
    return np.einsum(
        "ij,kjl,lm->kim",
        T_object_centered,
        poses,
        T_canonical_tcp,
    )


# Kept as a source-compatible alias for artifacts/scripts written before the
# sampler-frame contract was made explicit.  The old name was misleading:
# these inputs have always been GraspGenX canonical poses, not native Hand-E
# base_link poses.
gripper_base_poses_to_object_tcp = canonical_grasp_poses_to_object_tcp


def load_hande_dimensions(
    config_path: str | Path,
    *,
    tcp_offset_override_m: float | None = None,
    pregrasp_width_override_m: float | None = None,
) -> tuple[float, float, dict[str, str]]:
    """Load Hand-E TCP depth and total opening from its GraspGenX config.

    The Hand-E config stores one symmetric finger displacement in ``open``;
    total jaw opening is therefore twice that joint value.
    """

    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    fingertip = np.asarray(payload.get("fingertip"), dtype=float)
    if fingertip.shape != (3,) or not np.all(np.isfinite(fingertip)):
        raise ValueError(f"invalid fingertip in Hand-E config: {config_path}")
    open_joints = payload.get("open")
    if not isinstance(open_joints, dict) or len(open_joints) != 1:
        raise ValueError(f"Hand-E config must define one symmetric open joint: {config_path}")
    open_joint_m = float(next(iter(open_joints.values())))

    tcp_offset_m = (
        float(tcp_offset_override_m) if tcp_offset_override_m is not None else float(fingertip[2])
    )
    pregrasp_width_m = (
        float(pregrasp_width_override_m)
        if pregrasp_width_override_m is not None
        else 2.0 * open_joint_m
    )
    configured_full_open_m = 2.0 * open_joint_m
    if not np.isfinite(tcp_offset_m) or tcp_offset_m <= 0.0:
        raise ValueError("Hand-E TCP offset must be finite and positive")
    if not np.isfinite(pregrasp_width_m) or pregrasp_width_m <= 0.0:
        raise ValueError("Hand-E pregrasp width must be finite and positive")
    if pregrasp_width_m > configured_full_open_m + 1e-12:
        raise ValueError(
            "Hand-E pregrasp width exceeds the configured full opening "
            f"({configured_full_open_m:.6f} m)"
        )

    sources = {
        "tcp_offset": (
            "cli_override" if tcp_offset_override_m is not None else "config.fingertip[2]"
        ),
        "pregrasp_width": (
            "cli_override"
            if pregrasp_width_override_m is not None
            else "2 * config.open symmetric finger joint"
        ),
    }
    return tcp_offset_m, pregrasp_width_m, sources


def save_candidates_npz(
    output_path: str | Path,
    poses: np.ndarray,
    scores: np.ndarray,
    pregrasp_width_m: float,
    *,
    object_frame_id: str,
    base_to_tcp_offset_m: float,
) -> None:
    """Write the canonical candidate NPZ consumed by ``select-grasp``."""

    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("candidate output path must end in .npz")
    poses = np.asarray(poses, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must have shape (K, 4, 4), got {poses.shape}")
    if scores.shape != (len(poses),):
        raise ValueError(f"scores must have shape ({len(poses)},), got {scores.shape}")
    if len(poses) == 0:
        raise ValueError("cannot save an empty candidate set")
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(scores)):
        raise ValueError("candidate poses and scores must be finite")
    if not np.isfinite(pregrasp_width_m) or pregrasp_width_m <= 0.0:
        raise ValueError("pregrasp_width_m must be finite and positive")
    if not object_frame_id.strip():
        raise ValueError("object_frame_id must be non-empty")
    if not np.isfinite(base_to_tcp_offset_m) or base_to_tcp_offset_m <= 0.0:
        raise ValueError("base_to_tcp_offset_m must be finite and positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        poses=poses,
        scores=scores,
        widths_m=np.full(len(poses), pregrasp_width_m, dtype=np.float64),
        gripper_name=np.asarray(GRIPPER_NAME),
        object_frame_id=np.asarray(object_frame_id),
        base_to_tcp_offset_m=np.asarray(base_to_tcp_offset_m, dtype=np.float64),
        frame_convention=np.asarray(FRAME_CONVENTION),
    )


def save_provenance_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write human-readable candidate provenance."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
