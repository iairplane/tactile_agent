#!/usr/bin/env python3
"""Compile a reviewed pickup demonstration into a prototype agentic skill.

The compiler converts human motion into phase intent and records the selected
robot-specific object-frame grasp. Simplified human paths are retained for
diagnostics and preview only; the skill never exposes them as robot execution
inputs. The runtime agent plans free-space stages, while the selected GRASP SE(3)
endpoint remains fixed relative to the object.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo2gripper.task_skill import (
    AnchorFrame,
    FailurePolicy,
    ObjectGraspRelation,
    PlannerMode,
    SkillStage,
    SkillStatus,
    StageName,
    StageTargetPolicy,
    TaskSkill,
    WaypointRepresentation,
)

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

STAGE_COLORS_BGR = {
    "OBSERVE": (160, 160, 160),
    "APPROACH": (255, 170, 0),
    "PRESHAPE": (255, 230, 0),
    "GRASP": (0, 80, 255),
    "LIFT": (0, 210, 70),
    "TRANSPORT": (220, 80, 180),
    "PLACE": (255, 100, 80),
    "RELEASE": (40, 220, 255),
    "RETREAT": (200, 130, 255),
}

MOTION_STAGES = {"APPROACH", "PRESHAPE", "LIFT", "TRANSPORT", "PLACE", "RETREAT"}

SELECTED_GRASP_FRAME_CONVENTION = (
    "T_object_gripper_tcp; +X closing, +Z approach"
)
SELECTED_GRASP_GRIPPER_NAME = "robotiq_hande"
ROBOT_ADAPTED_INTENT_SCHEMA = "demo2gripper.robot_adapted_grasp_intent.v1"
ROBOT_ADAPTED_INTENT_ROLE = "robot_adapted_hande"
SELECTION_HARD_GATE_KEYS = (
    "min_gripper_width_m",
    "max_gripper_width_m",
    "max_center_distance_m",
    "max_closing_axis_error_rad",
    "max_approach_axis_error_rad",
)
HANDE_MAX_OPENING_M = 0.05


@dataclass(frozen=True)
class StageInterval:
    """Half-open demonstration interval for a semantic skill stage."""

    name: str
    start: int | None
    end: int | None
    enabled: bool = True
    evidence: str = "derived"

    @property
    def frame_count(self) -> int:
        if not self.enabled or self.start is None or self.end is None:
            return 0
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class SimplifiedPath:
    """Equal-arc waypoints sampled from an RDP-simplified human proxy path."""

    frames: np.ndarray
    timestamps_s: np.ndarray
    positions_base_m: np.ndarray
    path_length_m: float
    source_valid_frame_count: int
    rdp_vertex_count: int


@dataclass(frozen=True)
class AnchoredPath:
    """Position-only waypoints with an explicit runtime grounding reference."""

    stage: str
    path: SimplifiedPath
    anchor: AnchorFrame
    demonstrated_anchor_base_m: np.ndarray
    source_interval: tuple[int, int]
    boundary_constraints: tuple[str, ...] = ()

    @property
    def frames(self) -> np.ndarray:
        return self.path.frames

    @property
    def timestamps_s(self) -> np.ndarray:
        return self.path.timestamps_s

    @property
    def positions_base_m(self) -> np.ndarray:
        return self.path.positions_base_m

    @property
    def position_offsets_base_m(self) -> np.ndarray:
        return self.positions_base_m - self.demonstrated_anchor_base_m

    @property
    def path_length_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.positions_base_m, axis=0), axis=1).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--demo-dir",
        required=True,
        type=Path,
        help="Demonstration root containing input/, hand_pose/, and pickup_result_final/",
    )
    parser.add_argument(
        "--selected-grasp",
        required=True,
        type=Path,
        help=(
            "Selected Robotiq Hand-E grasp JSON containing a rigid "
            "T_object_gripper_tcp"
        ),
    )
    parser.add_argument(
        "--pickup-result-dir",
        type=Path,
        help="Override the pickup result directory",
    )
    parser.add_argument("--input-dir", type=Path, help="Override the RGB-D input directory")
    parser.add_argument("--hand-pose-dir", type=Path, help="Override the hand-pose directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (defaults to <demo-dir>/agentic_skill_prototype)",
    )
    parser.add_argument("--min-joint-confidence", type=float, default=0.5)
    parser.add_argument("--hand-presence-confidence", type=float, default=0.05)
    parser.add_argument("--approach-reliability-window-frames", type=int, default=10)
    parser.add_argument("--approach-reliability-fraction", type=float, default=0.5)
    parser.add_argument("--rdp-epsilon-m", type=float, default=0.002)
    parser.add_argument("--min-waypoints", type=int, default=4)
    parser.add_argument("--max-waypoints", type=int, default=24)
    parser.add_argument("--smoothing-window-frames", type=int, default=5)
    parser.add_argument("--release-opening-delta-m", type=float, default=0.010)
    parser.add_argument("--release-retreat-distance-m", type=float, default=0.030)
    parser.add_argument("--retreat-clearance-m", type=float, default=0.120)
    parser.add_argument("--min-retreat-lift-m", type=float, default=0.020)
    parser.add_argument(
        "--max-prelift-horizontal-m",
        type=float,
        default=0.005,
        help="Maximum horizontal retreat before the minimum vertical lift is reached",
    )
    parser.add_argument("--min-support-clearance-m", type=float, default=0.003)
    parser.add_argument("--max-stage-continuity-gap-m", type=float, default=1e-6)
    parser.add_argument("--transport-min-horizontal-m", type=float, default=0.015)
    parser.add_argument("--transport-height-fraction", type=float, default=0.80)
    parser.add_argument("--video-fps", type=float, help="Override preview video FPS")
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise ValueError(f"required file does not exist: {path}")
    return path


def _numeric_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"RGB directory does not exist: {directory}")
    paths = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    try:
        paths.sort(key=lambda path: int(path.stem))
    except ValueError as exc:
        raise ValueError(f"RGB filenames must have integer stems: {directory}") from exc
    if not paths:
        raise ValueError(f"no RGB frames found in {directory}")
    return paths


def _load_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(transform[:3, :3]) < 0.999:
        raise ValueError(f"{name} rotation is not right-handed")
    return transform


def _load_selected_grasp(
    path: Path,
) -> tuple[dict[str, Any], np.ndarray, float, float]:
    """Load the robot-specific selection without accepting ambiguous frames."""

    _require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selected grasp JSON root must be an object")
    if payload.get("frame_convention") != SELECTED_GRASP_FRAME_CONVENTION:
        raise ValueError(
            "selected grasp frame_convention must be "
            f"{SELECTED_GRASP_FRAME_CONVENTION!r}"
        )
    object_frame_id = payload.get("object_frame_id")
    if not isinstance(object_frame_id, str) or not object_frame_id.strip():
        raise ValueError("selected grasp object_frame_id must be a non-empty string")
    if payload.get("gripper_name") != SELECTED_GRASP_GRIPPER_NAME:
        raise ValueError("selected grasp gripper_name must be 'robotiq_hande'")

    intent_provenance = payload.get("intent_provenance")
    if not isinstance(intent_provenance, dict):
        raise ValueError(
            "Hand-E selected grasp must include robot-adapted intent_provenance"
        )
    if intent_provenance.get("schema") != ROBOT_ADAPTED_INTENT_SCHEMA:
        raise ValueError(
            "Hand-E selected grasp intent_provenance.schema must be "
            f"{ROBOT_ADAPTED_INTENT_SCHEMA!r}"
        )
    if intent_provenance.get("intent_role") != ROBOT_ADAPTED_INTENT_ROLE:
        raise ValueError(
            "Hand-E selected grasp intent_provenance.intent_role must be "
            f"{ROBOT_ADAPTED_INTENT_ROLE!r}"
        )

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Hand-E selected grasp must include selection evidence")
    if selection.get("hard_gates_passed") is not True:
        raise ValueError("Hand-E selected grasp hard_gates_passed must be true")
    thresholds = selection.get("hard_gate_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(
            "Hand-E selected grasp must include selection hard_gate_thresholds"
        )
    parsed_thresholds: dict[str, float] = {}
    for key in SELECTION_HARD_GATE_KEYS:
        try:
            value = float(thresholds[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"selected grasp hard gate {key} must be finite"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"selected grasp hard gate {key} must be finite")
        parsed_thresholds[key] = value
    if parsed_thresholds["min_gripper_width_m"] < 0.0:
        raise ValueError("selected grasp minimum gripper width must be non-negative")
    if (
        parsed_thresholds["max_gripper_width_m"]
        < parsed_thresholds["min_gripper_width_m"]
    ):
        raise ValueError("selected grasp gripper width hard gates are inconsistent")
    for key in (
        "max_center_distance_m",
        "max_closing_axis_error_rad",
        "max_approach_axis_error_rad",
    ):
        if parsed_thresholds[key] < 0.0:
            raise ValueError(f"selected grasp hard gate {key} must be non-negative")

    transform = _load_transform(
        payload.get("T_object_gripper_tcp"), "T_object_gripper_tcp"
    )
    try:
        pregrasp_width_m = float(payload["pregrasp_width_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selected grasp pregrasp_width_m must be finite") from exc
    if (
        not np.isfinite(pregrasp_width_m)
        or pregrasp_width_m <= 0.0
        or pregrasp_width_m > HANDE_MAX_OPENING_M
    ):
        raise ValueError(
            "selected grasp pregrasp_width_m must lie in (0, 0.05] for Hand-E"
        )
    try:
        approach_distance_m = float(payload["approach_distance_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selected grasp approach_distance_m must be finite") from exc
    if not np.isfinite(approach_distance_m) or approach_distance_m <= 0.0:
        raise ValueError("selected grasp approach_distance_m must be positive and finite")

    measured_hard_gates = {
        "min_gripper_width_m": pregrasp_width_m
        >= parsed_thresholds["min_gripper_width_m"],
        "max_gripper_width_m": pregrasp_width_m
        <= parsed_thresholds["max_gripper_width_m"],
    }
    for metric, threshold_key in (
        ("center_error_m", "max_center_distance_m"),
        ("closing_axis_error_rad", "max_closing_axis_error_rad"),
        ("approach_axis_error_rad", "max_approach_axis_error_rad"),
    ):
        try:
            measured = float(payload[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"selected grasp {metric} must be finite") from exc
        if not np.isfinite(measured):
            raise ValueError(f"selected grasp {metric} must be finite")
        measured_hard_gates[threshold_key] = measured <= parsed_thresholds[threshold_key]
    failed_hard_gates = [
        key for key, passed in measured_hard_gates.items() if not passed
    ]
    if failed_hard_gates:
        raise ValueError(
            "selected grasp measurements violate recorded hard gates: "
            + ", ".join(failed_hard_gates)
        )
    return payload, transform, pregrasp_width_m, approach_distance_m


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]


def _transform_vectors(transform: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=float) @ transform[:3, :3].T


def _validate_array_shape(array: np.ndarray, shape: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {array.shape}")
    for actual, expected in zip(array.shape, shape, strict=True):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")


def _first_sustained(mask: np.ndarray, start: int, count: int) -> int | None:
    values = np.asarray(mask, dtype=bool)
    for index in range(max(0, start), max(0, len(values) - count + 1)):
        if np.all(values[index : index + count]):
            return index
    return None


def _first_reliable_window(
    mask: np.ndarray,
    start: int,
    window: int,
    fraction: float,
) -> int | None:
    """Find the first valid sample followed by sustained, possibly intermittent evidence."""

    values = np.asarray(mask, dtype=bool)
    if window <= 0:
        raise ValueError("reliability window must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("reliability fraction must lie in (0, 1]")
    required = int(np.ceil(window * fraction))
    for index in range(max(0, start), max(0, len(values) - window + 1)):
        if values[index] and np.count_nonzero(values[index : index + window]) >= required:
            return index
    return None


def _contiguous_run_around(mask: np.ndarray, index: int) -> tuple[int, int]:
    values = np.asarray(mask, dtype=bool)
    if not 0 <= index < len(values) or not values[index]:
        return index, index + 1
    start = index
    end = index + 1
    while start > 0 and values[start - 1]:
        start -= 1
    while end < len(values) and values[end]:
        end += 1
    return start, end


def _infer_stage_intervals(
    pickup: dict[str, Any],
    track: dict[str, np.ndarray],
    joints_camera_m: np.ndarray,
    confidence: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[StageInterval], dict[str, Any]]:
    frame_count = len(confidence)
    near_approach_start, approach_end = map(
        int, pickup["approach_trajectory_interval"]
    )
    contact_start, contact_end = map(int, pickup["stable_contact_interval"])
    lift_onset = int(pickup["lift_onset_frame"])
    peak = int(pickup["peak_lift_frame"])
    airborne_start, airborne_end = map(int, pickup["airborne_interval"])

    ordered = [
        near_approach_start,
        approach_end,
        contact_start,
        contact_end,
        lift_onset,
        peak,
    ]
    if ordered != sorted(ordered) or not 0 <= near_approach_start < peak < frame_count:
        raise ValueError(f"pickup event frames are not ordered: {ordered}")

    vertical = np.asarray(track["vertical_displacement_m"], dtype=float)
    object_centers = np.asarray(track["object_centers_base_m"], dtype=float)
    pinch_centers = np.asarray(track["pinch_centers_base_m"], dtype=float)
    pinch_confidence = np.minimum(confidence[:, THUMB_TIP], confidence[:, INDEX_TIP])
    pinch_reliable = (
        (pinch_confidence >= args.min_joint_confidence)
        & np.isfinite(joints_camera_m[:, [THUMB_TIP, INDEX_TIP]]).all(axis=(1, 2))
    )
    reference_prefix = int(pickup["validation_config"]["reference_prefix_frames"])
    first_reliable = _first_reliable_window(
        pinch_reliable,
        reference_prefix,
        args.approach_reliability_window_frames,
        args.approach_reliability_fraction,
    )
    if first_reliable is None:
        raise ValueError("no sustained reliable pinch track after the reference prefix")
    approach_start = max(reference_prefix, first_reliable)
    if approach_start >= approach_end:
        raise ValueError("reliable approach track begins after the fitted approach endpoint")
    pinch_width = np.linalg.norm(
        joints_camera_m[:, INDEX_TIP] - joints_camera_m[:, THUMB_TIP], axis=1
    )
    pinch_object_distance = np.linalg.norm(pinch_centers - object_centers, axis=1)

    contact_slice = slice(contact_start, contact_end)
    contact_valid = pinch_confidence[contact_slice] >= args.min_joint_confidence
    if not np.any(contact_valid):
        raise ValueError("stable contact interval has no confident fingertip frames")
    grasp_opening = float(np.median(pinch_width[contact_slice][contact_valid]))
    grasp_distance = float(np.median(pinch_object_distance[contact_slice][contact_valid]))

    effective_settle_height = float(
        pickup.get(
            "quality", {}
        ).get("effective_lift_onset_threshold_m", pickup["validation_config"]["min_lift_onset_m"])
    )
    settled_mask = np.abs(vertical) <= effective_settle_height
    settled_start = _first_sustained(settled_mask, max(peak + 1, airborne_end - 1), 5)
    if settled_start is None:
        settled_start = min(frame_count - 1, max(peak + 1, airborne_end))

    release_width_threshold = grasp_opening + max(
        args.release_opening_delta_m, 0.25 * grasp_opening
    )
    release_open_mask = (pinch_confidence >= args.min_joint_confidence) & (
        pinch_width >= release_width_threshold
    )
    release_start = _first_sustained(release_open_mask, settled_start, 2)
    if release_start is None:
        release_start = min(frame_count - 2, settled_start + 10)

    retreat_distance_threshold = grasp_distance + args.release_retreat_distance_m
    retreat_mask = (pinch_confidence >= args.min_joint_confidence) & (
        pinch_object_distance >= retreat_distance_threshold
    )
    release_end = _first_sustained(retreat_mask, release_start + 1, 2)
    if release_end is None:
        release_end = min(frame_count - 1, release_start + 8)

    hand_presence = np.max(confidence, axis=1) >= args.hand_presence_confidence
    hand_end = None
    for index in range(release_end + 1, max(release_end + 1, frame_count - 4)):
        if not np.any(hand_presence[index : index + 5]):
            hand_end = index
            break
    if hand_end is None:
        present = np.flatnonzero(hand_presence[release_end:])
        hand_end = release_end + int(present[-1]) + 1 if len(present) else release_end + 1
    hand_end = min(frame_count, max(hand_end, release_end + 1))
    retreat_clear_mask = (pinch_confidence >= args.min_joint_confidence) & (
        pinch_object_distance >= args.retreat_clearance_m
    )
    retreat_clear_start = _first_sustained(retreat_clear_mask, release_end, 2)
    retreat_end = (
        min(hand_end, retreat_clear_start + 2)
        if retreat_clear_start is not None
        else hand_end
    )
    retreat_end = max(release_end + 1, retreat_end)

    peak_lift_m = float(pickup["peak_lift_m"])
    high_mask = vertical >= args.transport_height_fraction * peak_lift_m
    high_start, high_end = _contiguous_run_around(high_mask, peak)
    horizontal_motion = float(
        np.linalg.norm(object_centers[high_end - 1, :2] - object_centers[high_start, :2])
    )
    transport_enabled = (
        high_end - high_start >= 5
        and horizontal_motion >= args.transport_min_horizontal_m
        and high_start > contact_end
        and high_end < release_start
    )
    if transport_enabled:
        lift_end = high_start
        transport = StageInterval(
            "TRANSPORT", high_start, high_end, evidence="high-clearance horizontal motion"
        )
        place_start = high_end
    else:
        lift_end = peak + 1
        transport = StageInterval(
            "TRANSPORT",
            None,
            None,
            enabled=False,
            evidence="no distinct high-clearance horizontal segment",
        )
        place_start = peak + 1

    intervals = [
        StageInterval("OBSERVE", 0, approach_start, evidence="pre-action scene observation"),
        StageInterval(
            "APPROACH",
            approach_start,
            approach_end,
            evidence="reliable full approach through the near-contact fit",
        ),
        StageInterval("PRESHAPE", approach_end, contact_start, evidence="closing before contact"),
        StageInterval("GRASP", contact_start, contact_end, evidence="stable contact interval"),
        StageInterval(
            "LIFT",
            contact_end,
            lift_end,
            evidence="grasp-to-lift transition and sustained vertical object motion",
        ),
        transport,
        StageInterval("PLACE", place_start, release_start, evidence="descent and support settle"),
        StageInterval("RELEASE", release_start, release_end, evidence="finger opening"),
        StageInterval(
            "RETREAT", release_end, retreat_end, evidence="hand/object clearance reached"
        ),
    ]
    diagnostics = {
        "reference_prefix_frames": reference_prefix,
        "first_reliable_pinch_frame": first_reliable,
        "approach_near_fit_interval": [near_approach_start, approach_end],
        "compiled_approach_interval": [approach_start, approach_end],
        "contact_end_frame": contact_end,
        "pickup_lift_onset_frame": lift_onset,
        "compiled_lift_start_frame": contact_end,
        "airborne_interval": [airborne_start, airborne_end],
        "support_settled_frame": settled_start,
        "demonstrated_grasp_opening_m": grasp_opening,
        "release_opening_threshold_m": release_width_threshold,
        "demonstrated_grasp_center_distance_m": grasp_distance,
        "retreat_distance_threshold_m": retreat_distance_threshold,
        "high_clearance_interval": [high_start, high_end],
        "high_clearance_horizontal_motion_m": horizontal_motion,
        "transport_detected": transport_enabled,
        "retreat_clearance_m": args.retreat_clearance_m,
        "retreat_clearance_frame": retreat_end - 1,
        "hand_track_end_frame": hand_end,
    }
    return intervals, diagnostics


def _moving_median(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(points) <= 2:
        return np.asarray(points, dtype=float).copy()
    radius = window // 2
    result = np.empty_like(points, dtype=float)
    for index in range(len(points)):
        result[index] = np.median(
            points[max(0, index - radius) : min(len(points), index + radius + 1)], axis=0
        )
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def _point_segment_distances(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    length_sq = float(segment @ segment)
    if length_sq <= 1e-16:
        return np.linalg.norm(points - start, axis=1)
    amount = np.clip(((points - start) @ segment) / length_sq, 0.0, 1.0)
    projection = start + amount[:, None] * segment
    return np.linalg.norm(points - projection, axis=1)


def _rdp_indices(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) <= 2:
        return np.arange(len(points), dtype=int)
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    pending = [(0, len(points) - 1)]
    while pending:
        start, end = pending.pop()
        if end <= start + 1:
            continue
        distances = _point_segment_distances(points[start + 1 : end], points[start], points[end])
        local = int(np.argmax(distances))
        if distances[local] > epsilon:
            split = start + 1 + local
            keep[split] = True
            pending.extend(((start, split), (split, end)))
    return np.flatnonzero(keep)


def _equal_arc_resample(
    points: np.ndarray, timestamps_s: np.ndarray, frames: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        selected = np.linspace(0, len(points) - 1, count)
        positions = np.repeat(points[:1], count, axis=0)
        times = np.interp(selected, np.arange(len(points)), timestamps_s)
        output_frames = np.rint(np.interp(selected, np.arange(len(points)), frames)).astype(int)
        return positions, times, output_frames, 0.0
    targets = np.linspace(0.0, total, count)
    positions = np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(3)]
    )
    times = np.interp(targets, cumulative, timestamps_s)
    output_frames = np.rint(np.interp(targets, cumulative, frames)).astype(int)
    return positions, times, output_frames, total


def _simplify_path(
    positions_base_m: np.ndarray,
    timestamps_s: np.ndarray,
    valid: np.ndarray,
    start: int,
    end: int,
    args: argparse.Namespace,
) -> SimplifiedPath | None:
    frames = np.flatnonzero(valid[start:end]) + start
    if len(frames) < 2:
        return None
    points = _moving_median(positions_base_m[frames], args.smoothing_window_frames)
    rdp = _rdp_indices(points, args.rdp_epsilon_m)
    rdp_points = points[rdp]
    rdp_times = timestamps_s[frames[rdp]]
    rdp_frames = frames[rdp]
    waypoint_count = min(
        args.max_waypoints,
        max(2, args.min_waypoints, len(rdp)),
    )
    output_points, output_times, output_frames, path_length = _equal_arc_resample(
        rdp_points, rdp_times, rdp_frames, waypoint_count
    )
    return SimplifiedPath(
        frames=output_frames,
        timestamps_s=output_times,
        positions_base_m=output_points,
        path_length_m=path_length,
        source_valid_frame_count=len(frames),
        rdp_vertex_count=len(rdp),
    )


def _force_path_boundary(
    path: SimplifiedPath,
    *,
    side: str,
    frame: int,
    timestamp_s: float,
    position_base_m: np.ndarray,
) -> SimplifiedPath:
    """Replace or append a semantic boundary without fabricating an orientation."""

    position = np.asarray(position_base_m, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("forced waypoint boundary must be a finite 3-vector")
    frames = path.frames.copy()
    times = path.timestamps_s.copy()
    positions = path.positions_base_m.copy()
    if side == "start":
        if frames[0] == frame:
            times[0] = timestamp_s
            positions[0] = position
        else:
            frames = np.concatenate(([frame], frames))
            times = np.concatenate(([timestamp_s], times))
            positions = np.vstack((position, positions))
    elif side == "end":
        if frames[-1] == frame:
            times[-1] = timestamp_s
            positions[-1] = position
        else:
            frames = np.concatenate((frames, [frame]))
            times = np.concatenate((times, [timestamp_s]))
            positions = np.vstack((positions, position))
    else:
        raise ValueError("forced waypoint side must be 'start' or 'end'")
    if np.any(np.diff(frames) < 0) or np.any(np.diff(times) < 0.0):
        raise ValueError("forced waypoint boundary violates temporal ordering")
    return SimplifiedPath(
        frames=frames,
        timestamps_s=times,
        positions_base_m=positions,
        path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        source_valid_frame_count=path.source_valid_frame_count,
        rdp_vertex_count=path.rdp_vertex_count,
    )


def _anchor_path(
    stage: str,
    path: SimplifiedPath,
    *,
    anchor: AnchorFrame,
    demonstrated_anchor_base_m: np.ndarray,
    source_interval: tuple[int, int],
    boundary_constraints: tuple[str, ...] = (),
) -> AnchoredPath:
    anchor_position = np.asarray(demonstrated_anchor_base_m, dtype=float)
    if anchor_position.shape != (3,) or not np.isfinite(anchor_position).all():
        raise ValueError(f"{stage} anchor must be a finite 3-vector")
    return AnchoredPath(
        stage=stage,
        path=path,
        anchor=anchor,
        demonstrated_anchor_base_m=anchor_position,
        source_interval=source_interval,
        boundary_constraints=boundary_constraints,
    )


def _validate_compiled_paths(
    paths: dict[str, AnchoredPath],
    *,
    support_height_base_m: float,
    min_support_clearance_m: float,
    max_continuity_gap_m: float,
    min_retreat_lift_m: float,
    max_prelift_horizontal_m: float,
) -> dict[str, Any]:
    """Fail closed on broken joins, support intrusion, and unsafe retreat order."""

    scalar_limits = {
        "support_height_base_m": support_height_base_m,
        "min_support_clearance_m": min_support_clearance_m,
        "max_continuity_gap_m": max_continuity_gap_m,
        "min_retreat_lift_m": min_retreat_lift_m,
        "max_prelift_horizontal_m": max_prelift_horizontal_m,
    }
    for name, value in scalar_limits.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name != "support_height_base_m" and value < 0.0:
            raise ValueError(f"{name} must be non-negative")

    clearance_by_stage: dict[str, float] = {}
    for name, path in paths.items():
        positions = path.positions_base_m
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError(f"{name} waypoints must be a finite (N, 3) array")
        clearance = float(np.min(positions[:, 2] - support_height_base_m))
        clearance_by_stage[name] = clearance
        if clearance < min_support_clearance_m:
            raise ValueError(
                f"{name} minimum support clearance {clearance * 1000.0:.2f} mm is below "
                f"the required {min_support_clearance_m * 1000.0:.2f} mm"
            )

    continuity_pairs = [
        ("APPROACH", "PRESHAPE"),
        ("PRESHAPE", "LIFT"),
        ("LIFT", "TRANSPORT" if "TRANSPORT" in paths else "PLACE"),
    ]
    if "TRANSPORT" in paths:
        continuity_pairs.append(("TRANSPORT", "PLACE"))
    continuity_pairs.append(("PLACE", "RETREAT"))
    continuity_gaps: dict[str, float] = {}
    for before, after in continuity_pairs:
        if before not in paths or after not in paths:
            continue
        gap = float(
            np.linalg.norm(
                paths[before].positions_base_m[-1] - paths[after].positions_base_m[0]
            )
        )
        continuity_gaps[f"{before}->{after}"] = gap
        if gap > max_continuity_gap_m:
            raise ValueError(
                f"{before}->{after} waypoint discontinuity is {gap * 1000.0:.3f} mm"
            )

    retreat = paths.get("RETREAT")
    if retreat is None:
        raise ValueError("RETREAT requires a compiled waypoint path")
    retreat_offsets = retreat.position_offsets_base_m
    if not np.isfinite(retreat_offsets).all():
        raise ValueError("RETREAT offsets from the release-pose anchor must be finite")
    if float(np.linalg.norm(retreat_offsets[0])) > max_continuity_gap_m:
        raise ValueError("RETREAT must start at its release-pose anchor")

    retreat_vertical = retreat_offsets[:, 2]
    retreat_peak_lift = float(np.max(retreat_vertical))
    retreat_min_vertical = float(np.min(retreat_vertical))
    if retreat_min_vertical < -max_continuity_gap_m:
        raise ValueError(
            "RETREAT drops below its release-pose anchor before clearing the object: "
            f"{retreat_min_vertical * 1000.0:.2f} mm"
        )
    if retreat_peak_lift < min_retreat_lift_m:
        raise ValueError(
            f"RETREAT lift {retreat_peak_lift * 1000.0:.1f} mm is below the required "
            f"{min_retreat_lift_m * 1000.0:.1f} mm"
        )

    crossing_horizontal = 0.0
    max_horizontal_before_clearance = 0.0
    if min_retreat_lift_m > 0.0:
        crossing_index = int(np.flatnonzero(retreat_vertical >= min_retreat_lift_m)[0])
        if np.any(
            np.diff(retreat_vertical[: crossing_index + 1]) < -max_continuity_gap_m
        ):
            raise ValueError("RETREAT must rise monotonically until lift clearance is reached")

        if crossing_index == 0:
            crossing_xy = retreat_offsets[0, :2]
            pre_clearance_xy = retreat_offsets[:1, :2]
        else:
            before = crossing_index - 1
            vertical_delta = retreat_vertical[crossing_index] - retreat_vertical[before]
            alpha = (
                (min_retreat_lift_m - retreat_vertical[before]) / vertical_delta
            )
            crossing_xy = (
                retreat_offsets[before, :2]
                + alpha
                * (retreat_offsets[crossing_index, :2] - retreat_offsets[before, :2])
            )
            pre_clearance_xy = np.vstack(
                (retreat_offsets[:crossing_index, :2], crossing_xy)
            )
        crossing_horizontal = float(np.linalg.norm(crossing_xy))
        max_horizontal_before_clearance = float(
            np.max(np.linalg.norm(pre_clearance_xy, axis=1))
        )
        if max_horizontal_before_clearance > max_prelift_horizontal_m:
            raise ValueError(
                "RETREAT moves horizontally before reaching lift clearance: "
                f"{max_horizontal_before_clearance * 1000.0:.2f} mm exceeds "
                f"{max_prelift_horizontal_m * 1000.0:.2f} mm"
            )
    return {
        "minimum_support_clearance_m_by_stage": clearance_by_stage,
        "continuity_gap_m_by_transition": continuity_gaps,
        "maximum_continuity_gap_m": max(continuity_gaps.values(), default=0.0),
        "retreat_min_vertical_offset_m": retreat_min_vertical,
        "retreat_peak_vertical_offset_m": retreat_peak_lift,
        "retreat_clearance_crossing_horizontal_offset_m": crossing_horizontal,
        "retreat_max_horizontal_offset_before_clearance_m": (
            max_horizontal_before_clearance
        ),
    }


def _stage_contract(
    stage: StageInterval,
    has_path: bool,
    grounding_anchor: AnchorFrame | None = None,
) -> dict[str, Any]:
    shared: dict[str, Any] = {
        "name": stage.name,
        "enabled": stage.enabled,
        "demonstration_interval": (
            [stage.start, stage.end] if stage.start is not None and stage.end is not None else None
        ),
        "demonstration_frame_count": stage.frame_count,
        "evidence": stage.evidence,
        "execution_authorized": False,
        "grounding_anchor": (
            None if grounding_anchor is None else grounding_anchor.value
        ),
    }
    contracts = {
        "OBSERVE": {
            "intent": "localize the current object and refresh the planning scene",
            "action": "perception_gate",
            "exit_guard": "fresh object observation with acceptable confidence",
            "on_failure": "hold and reacquire perception",
        },
        "APPROACH": {
            "intent": "reach an agent-selected pregrasp for the grounded side-pinch constraint",
            "action": "motion_plan",
            "grounding": "agent_generated_from_current_scene_and_grounded_grasp",
            "orientation_policy": "agent selects a robot-feasible approach and TCP pose",
            "exit_guard": "planned pregrasp region reached",
            "on_failure": "reobserve object and replan",
        },
        "PRESHAPE": {
            "intent": "resolve the grounded jaw line into a robot-specific preshape",
            "action": "motion_plan_and_gripper_preshape",
            "grounding": "agent_generated_from_grounded_side_pinch_constraint",
            "orientation_policy": "preserve the demonstrated jaw line; select approach online",
            "exit_guard": "preshape pose reached and object observation is fresh",
            "on_failure": "hold or retreat, reobserve, and generate another candidate",
        },
        "GRASP": {
            "intent": "close around the object at the selected Hand-E SE(3) grasp",
            "action": "gripper_close",
            "grounding": "selected_object_relative_se3",
            "orientation_policy": "pinch center plus undirected horizontal jaw line",
            "exit_guard": "gripper/contact evidence confirms acquisition",
            "on_failure": "open, retreat, reobserve, and retry",
        },
        "LIFT": {
            "intent": "clear the support while preserving the verified grasp",
            "action": "motion_plan",
            "grounding": "agent_generated_from_measured_post_grasp_state",
            "orientation_policy": "hold grasp orientation",
            "exit_guard": "object remains coupled and lift clearance is reached",
            "on_failure": "stop, lower if safe, then request recovery",
        },
        "TRANSPORT": {
            "intent": "move the grasped object toward the task-selected placement region",
            "action": "motion_plan",
            "grounding": "agent_generated_from_current_scene_and_task_goal",
            "orientation_policy": "hold grasp orientation",
            "exit_guard": "transport endpoint reached with object coupled",
            "on_failure": "hold and replan around the current planning scene",
        },
        "PLACE": {
            "intent": "reach a feasible placement selected from task intent and scene state",
            "action": "motion_plan",
            "grounding": "agent_generated_from_placement_policy",
            "orientation_policy": "hold grasp orientation",
            "exit_guard": "support/contact evidence confirms placement",
            "on_failure": "hold object and replan the placement approach",
        },
        "RELEASE": {
            "intent": "open the gripper after placement is verified",
            "action": "gripper_open",
            "exit_guard": "opening reached and object remains on support",
            "on_failure": "hold position and retry gripper action",
        },
        "RETREAT": {
            "intent": "create clearance from the measured release pose",
            "action": "motion_plan",
            "grounding": "agent_generated_from_measured_release_state",
            "orientation_policy": "planner_selected",
            "exit_guard": "retreat clearance reached",
            "on_failure": "replan from current state",
        },
    }
    shared.update(contracts[stage.name])
    shared["runtime_target_policy"] = (
        None
        if stage.name == "OBSERVE"
        else (
            StageTargetPolicy.OBJECT_RELATIVE_GRASP.value
            if stage.name == "GRASP"
            else (
                StageTargetPolicy.TASK_INTENT.value
                if stage.name == "RELEASE"
                else StageTargetPolicy.ONLINE_AGENT_PLAN.value
            )
        )
    )
    shared["trajectory_available"] = has_path
    shared["trajectory_npz_prefix"] = stage.name.lower() if has_path else None
    shared["demonstration_path_role"] = "diagnostic_only" if has_path else None
    shared["demonstration_path_consumed_for_execution"] = False
    return shared


def _task_skill_stage(
    stage: StageInterval,
    timestamps_s: np.ndarray,
    path: AnchoredPath | None,
) -> SkillStage:
    """Convert an inferred interval into the v3 online-planning contract.

    ``path`` is accepted only to preserve its diagnostic provenance. It never
    becomes a :class:`SkillStage` waypoint reference.
    """

    if stage.start is None or stage.end is None or not stage.enabled:
        raise ValueError("cannot compile a disabled or empty stage")
    name = StageName(stage.name)
    relative_start = float(timestamps_s[stage.start] - timestamps_s[0])
    if stage.end < len(timestamps_s):
        relative_end = float(timestamps_s[stage.end] - timestamps_s[0])
    else:
        relative_end = float(
            timestamps_s[-1] - timestamps_s[0] + np.median(np.diff(timestamps_s))
        )
    conditions = {
        StageName.APPROACH: (
            ("object_observation_fresh", "planning_scene_current"),
            ("preshape_entry_reached",),
            FailurePolicy.REOBSERVE_REPLAN,
        ),
        StageName.PRESHAPE: (
            ("object_observation_fresh", "approach_complete"),
            ("grasp_translation_reached", "gripper_opening_ready"),
            FailurePolicy.RETREAT_REOBSERVE_RETRY,
        ),
        StageName.GRASP: (
            ("preshape_complete", "grasp_translation_grounded"),
            ("grasp_acquisition_verified",),
            FailurePolicy.RETREAT_REOBSERVE_RETRY,
        ),
        StageName.LIFT: (
            ("grasp_acquisition_verified",),
            ("lift_clearance_reached", "object_remains_coupled"),
            FailurePolicy.HOLD,
        ),
        StageName.TRANSPORT: (
            ("lift_complete", "object_remains_coupled"),
            ("transport_endpoint_reached", "object_remains_coupled"),
            FailurePolicy.REOBSERVE_REPLAN,
        ),
        StageName.PLACE: (
            ("object_remains_coupled", "placement_region_available"),
            ("placement_support_verified",),
            FailurePolicy.HOLD,
        ),
        StageName.RELEASE: (
            ("placement_support_verified",),
            ("gripper_open_verified", "object_stays_placed"),
            FailurePolicy.HOLD,
        ),
        StageName.RETREAT: (
            ("release_complete", "object_stays_placed"),
            ("retreat_clearance_reached",),
            FailurePolicy.REOBSERVE_REPLAN,
        ),
    }
    preconditions, completion, failure_policy = conditions[name]
    del path
    gripper_action = name in {StageName.GRASP, StageName.RELEASE}
    anchors = {
        StageName.APPROACH: AnchorFrame.GRASP_TRANSLATION,
        StageName.PRESHAPE: AnchorFrame.GRASP_TRANSLATION,
        StageName.GRASP: AnchorFrame.GRASP_TRANSLATION,
        StageName.LIFT: AnchorFrame.RUNTIME_STAGE_START,
        StageName.TRANSPORT: AnchorFrame.RUNTIME_STAGE_START,
        StageName.PLACE: AnchorFrame.PLACEMENT,
        StageName.RELEASE: AnchorFrame.RELEASE_POSE,
        StageName.RETREAT: AnchorFrame.RELEASE_POSE,
    }
    target_policy = (
        StageTargetPolicy.OBJECT_RELATIVE_GRASP
        if name is StageName.GRASP
        else (
            StageTargetPolicy.TASK_INTENT
            if name is StageName.RELEASE
            else StageTargetPolicy.ONLINE_AGENT_PLAN
        )
    )
    return SkillStage(
        name=name,
        frame_interval=(stage.start, stage.end),
        time_interval_s=(relative_start, relative_end),
        anchor=anchors[name],
        planner_mode=(
            PlannerMode.GRIPPER_ACTION if gripper_action else PlannerMode.MOTION_PLANNING
        ),
        target_policy=target_policy,
        waypoint_artifact=None,
        preconditions=preconditions,
        completion=completion,
        failure_policy=failure_policy,
        optional=name is StageName.TRANSPORT,
    )


def _phase_for_frame(intervals: list[StageInterval], frame: int) -> str:
    for interval in intervals:
        if (
            interval.enabled
            and interval.start is not None
            and interval.end is not None
            and interval.start <= frame < interval.end
        ):
            return interval.name
    return "UNASSIGNED"


def _draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    finite = np.isfinite(points).all(axis=1)
    for first, second, good in zip(points[:-1], points[1:], finite[:-1] & finite[1:], strict=True):
        if good:
            cv2.line(
                image,
                tuple(np.rint(first).astype(int)),
                tuple(np.rint(second).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )


def _project_plot(
    panel: np.ndarray,
    paths: dict[str, AnchoredPath],
    axes: tuple[int, int],
    rect: tuple[int, int, int, int],
    labels: tuple[str, str],
) -> None:
    x0, y0, x1, y1 = rect
    available = [
        path.positions_base_m[:, axes]
        for path in paths.values()
        if len(path.positions_base_m)
    ]
    cv2.rectangle(panel, (x0, y0), (x1, y1), (75, 75, 75), 1)
    cv2.putText(
        panel,
        labels[0],
        (x1 - 22, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
    )
    cv2.putText(
        panel,
        labels[1],
        (x0 + 8, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
    )
    if not available:
        return
    all_points = np.concatenate(available)
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    span = np.maximum(upper - lower, 0.005)
    lower -= 0.08 * span
    upper += 0.08 * span

    def convert(points: np.ndarray) -> np.ndarray:
        values = points[:, axes]
        px = x0 + (values[:, 0] - lower[0]) / (upper[0] - lower[0]) * (x1 - x0)
        py = y1 - (values[:, 1] - lower[1]) / (upper[1] - lower[1]) * (y1 - y0)
        return np.column_stack((px, py))

    for name, path in paths.items():
        points = convert(path.positions_base_m)
        color = STAGE_COLORS_BGR[name]
        _draw_polyline(panel, points, color, 2)
        for point in points:
            cv2.circle(panel, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)


def _write_static_preview(
    path: Path,
    rgb_paths: list[Path],
    keypoints_2d: np.ndarray,
    pinch_valid: np.ndarray,
    intervals: list[StageInterval],
    paths: dict[str, AnchoredPath],
    representative_frame: int,
    diagnostics: dict[str, Any],
) -> None:
    image = cv2.imread(str(rgb_paths[representative_frame]), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"could not read representative RGB frame: {rgb_paths[representative_frame]}"
        )
    height, width = image.shape[:2]
    panel_width = 560
    canvas = np.full((max(height, 720), width + panel_width, 3), 24, dtype=np.uint8)
    canvas[:height, :width] = image
    pinch_2d = np.mean(keypoints_2d[:, [THUMB_TIP, INDEX_TIP]], axis=1)
    for interval in intervals:
        if not interval.enabled or interval.name not in MOTION_STAGES:
            continue
        assert interval.start is not None and interval.end is not None
        source = paths.get(interval.name)
        source_start, source_end = (
            source.source_interval if source is not None else (interval.start, interval.end)
        )
        frames = np.arange(source_start, source_end)
        frames = frames[pinch_valid[frames]]
        if len(frames) >= 2:
            _draw_polyline(canvas, pinch_2d[frames], STAGE_COLORS_BGR[interval.name], 3)
            cv2.circle(
                canvas,
                tuple(np.rint(pinch_2d[frames[-1]]).astype(int)),
                5,
                STAGE_COLORS_BGR[interval.name],
                -1,
                cv2.LINE_AA,
            )
    cv2.rectangle(canvas, (0, 0), (width - 1, 58), (15, 15, 15), -1)
    cv2.putText(
        canvas,
        "AGENTIC SKILL PROTOTYPE - REVIEW ONLY",
        (18, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (50, 210, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Planar side-pinch constraint; human paths are diagnostics only",
        (18, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    x_text = width + 20
    cv2.putText(
        canvas,
        "Compiled stages",
        (x_text, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    y = 62
    for interval in intervals:
        color = STAGE_COLORS_BGR[interval.name]
        cv2.circle(canvas, (x_text + 7, y - 5), 5, color, -1, cv2.LINE_AA)
        if interval.enabled:
            label = f"{interval.name:<10} [{interval.start:03d}, {interval.end:03d})"
        else:
            label = f"{interval.name:<10} skipped"
        cv2.putText(
            canvas,
            label,
            (x_text + 20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        y += 25
    cv2.putText(
        canvas,
        f"transport XY: {diagnostics['high_clearance_horizontal_motion_m'] * 1000:.1f} mm",
        (x_text, y + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    _project_plot(
        canvas,
        paths,
        (0, 1),
        (width + 20, 340, width + panel_width - 20, 505),
        ("base X", "base Y"),
    )
    _project_plot(
        canvas,
        paths,
        (0, 2),
        (width + 20, 535, width + panel_width - 20, 700),
        ("base X", "base Z"),
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write preview: {path}")


def _write_video_preview(
    path: Path,
    rgb_paths: list[Path],
    keypoints_2d: np.ndarray,
    pinch_valid: np.ndarray,
    timestamps_s: np.ndarray,
    intervals: list[StageInterval],
    paths: dict[str, AnchoredPath],
    fps_override: float | None,
) -> float:
    first = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"could not read RGB frame: {rgb_paths[0]}")
    height, width = first.shape[:2]
    inferred_fps = float(1.0 / np.median(np.diff(timestamps_s)))
    fps = float(fps_override if fps_override is not None else inferred_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("preview video FPS must be finite and positive")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer for {path}")
    pinch_2d = np.mean(keypoints_2d[:, [THUMB_TIP, INDEX_TIP]], axis=1)
    try:
        for frame, rgb_path in enumerate(rgb_paths):
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"invalid RGB frame: {rgb_path}")
            stage = _phase_for_frame(intervals, frame)
            color = STAGE_COLORS_BGR.get(stage, (110, 110, 110))
            interval = next((item for item in intervals if item.name == stage), None)
            if interval is not None and interval.start is not None:
                path_data = paths.get(stage)
                trace_start = (
                    path_data.source_interval[0] if path_data is not None else interval.start
                )
                trace_frames = np.arange(
                    trace_start, min(frame + 1, interval.end or frame + 1)
                )
                trace_frames = trace_frames[pinch_valid[trace_frames]]
                if len(trace_frames) >= 2:
                    _draw_polyline(image, pinch_2d[trace_frames], color, 3)
            if pinch_valid[frame]:
                thumb = tuple(np.rint(keypoints_2d[frame, THUMB_TIP]).astype(int))
                index = tuple(np.rint(keypoints_2d[frame, INDEX_TIP]).astype(int))
                midpoint = tuple(np.rint(pinch_2d[frame]).astype(int))
                cv2.line(image, thumb, index, color, 2, cv2.LINE_AA)
                cv2.circle(image, midpoint, 5, color, -1, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (width, 70), (15, 15, 15), -1)
            cv2.putText(
                image,
                f"{stage}   frame {frame:03d}   t={timestamps_s[frame] - timestamps_s[0]:.2f}s",
                (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                "prototype / selected object-frame SE(3) / paths diagnostic only",
                (15, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"preview video was not written: {path}")
    return fps


def _path_json(path: AnchoredPath, npz_prefix: str) -> dict[str, Any]:
    return {
        "source": "human thumb/index pinch-center proxy in calibrated FR3 base frame",
        "simplification": "moving median, RDP, then equal-arc resampling",
        "source_frame_interval": list(path.source_interval),
        "source_valid_frame_count": path.path.source_valid_frame_count,
        "rdp_vertex_count": path.path.rdp_vertex_count,
        "waypoint_count": len(path.frames),
        "path_length_m": path.path_length_m,
        "anchor": path.anchor.value,
        "demonstrated_anchor_base_m": path.demonstrated_anchor_base_m.tolist(),
        "representation": WaypointRepresentation.POSITION_OFFSETS_BASE_M.value,
        "boundary_constraints": list(path.boundary_constraints),
        "orientation_resolved": False,
        "npz": {
            "frames": f"{npz_prefix}_frames",
            "position_offsets_base_m": f"{npz_prefix}_position_offsets_base_m",
        },
    }


def compile_skill(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path | None]:
    demo_dir = args.demo_dir
    pickup_dir = args.pickup_result_dir or demo_dir / "pickup_result_final"
    input_dir = args.input_dir or demo_dir / "input"
    hand_pose_dir = args.hand_pose_dir or demo_dir / "hand_pose"
    output_dir = args.output_dir or demo_dir / "agentic_skill_prototype"
    selected_grasp_path = args.selected_grasp.expanduser().resolve()
    (
        selected_grasp,
        T_object_gripper_tcp,
        selected_pregrasp_width_m,
        selected_approach_distance_m,
    ) = _load_selected_grasp(selected_grasp_path)

    finite_float_args = (
        "min_joint_confidence",
        "hand_presence_confidence",
        "rdp_epsilon_m",
        "release_opening_delta_m",
        "release_retreat_distance_m",
        "retreat_clearance_m",
        "min_retreat_lift_m",
        "max_prelift_horizontal_m",
        "min_support_clearance_m",
        "max_stage_continuity_gap_m",
        "transport_min_horizontal_m",
        "transport_height_fraction",
    )
    for name in finite_float_args:
        if not np.isfinite(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.video_fps is not None and not np.isfinite(args.video_fps):
        raise ValueError("--video-fps must be finite")

    if not 0.0 <= args.min_joint_confidence <= 1.0:
        raise ValueError("--min-joint-confidence must lie in [0, 1]")
    if not 0.0 <= args.hand_presence_confidence <= 1.0:
        raise ValueError("--hand-presence-confidence must lie in [0, 1]")
    if args.rdp_epsilon_m <= 0.0:
        raise ValueError("--rdp-epsilon-m must be positive")
    if args.retreat_clearance_m <= 0.0:
        raise ValueError("--retreat-clearance-m must be positive")
    if args.approach_reliability_window_frames <= 0:
        raise ValueError("--approach-reliability-window-frames must be positive")
    if not 0.0 < args.approach_reliability_fraction <= 1.0:
        raise ValueError("--approach-reliability-fraction must lie in (0, 1]")
    for name in (
        "min_retreat_lift_m",
        "max_prelift_horizontal_m",
        "min_support_clearance_m",
        "max_stage_continuity_gap_m",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 2 <= args.min_waypoints <= args.max_waypoints:
        raise ValueError("waypoint limits must satisfy 2 <= min <= max")
    if args.smoothing_window_frames <= 0:
        raise ValueError("--smoothing-window-frames must be positive")

    pickup_path = _require_file(pickup_dir / "pickup_grasp.json")
    track_path = _require_file(pickup_dir / "pickup_track.npz")
    joints_path = _require_file(hand_pose_dir / "hand_joints_camera_m.npy")
    confidence_path = _require_file(hand_pose_dir / "hand_joint_confidence.npy")
    keypoints_path = _require_file(hand_pose_dir / "hand_keypoints_2d.npy")
    timestamps_path = _require_file(input_dir / "timestamps_s.npy")
    rgb_paths = _numeric_image_paths(input_dir / "rgb")

    pickup = json.loads(pickup_path.read_text(encoding="utf-8"))
    if pickup.get("schema") != "demo2gripper.cad_free_pickup.v2":
        raise ValueError("pickup_grasp.json must use demo2gripper.cad_free_pickup.v2")
    if not pickup.get("pickup_demonstration_validated", False):
        raise ValueError("pickup demonstration is not validated")
    base_camera = _load_transform(pickup["T_base_camera"], "T_base_camera")
    demonstrated_grasp = _load_transform(
        pickup["T_base_demonstrated_grasp_frame"], "T_base_demonstrated_grasp_frame"
    )

    with np.load(track_path, allow_pickle=False) as archive:
        required_track = {
            "object_centers_base_m",
            "pinch_centers_base_m",
            "pinch_confidence",
            "vertical_displacement_m",
            "timestamps_s",
        }
        missing = required_track - set(archive.files)
        if missing:
            raise ValueError(f"pickup_track.npz is missing: {', '.join(sorted(missing))}")
        track = {key: np.asarray(archive[key]) for key in archive.files}
    joints_camera_m = np.asarray(np.load(joints_path, allow_pickle=False), dtype=float)
    confidence = np.asarray(np.load(confidence_path, allow_pickle=False), dtype=float)
    keypoints_2d = np.asarray(np.load(keypoints_path, allow_pickle=False), dtype=float)
    timestamps_s = np.asarray(np.load(timestamps_path, allow_pickle=False), dtype=float)
    frame_count = len(rgb_paths)
    _validate_array_shape(joints_camera_m, (frame_count, 21, 3), "hand joints")
    _validate_array_shape(confidence, (frame_count, 21), "hand confidence")
    _validate_array_shape(keypoints_2d, (frame_count, 21, 2), "2-D hand keypoints")
    _validate_array_shape(timestamps_s, (frame_count,), "timestamps")
    _validate_array_shape(track["object_centers_base_m"], (frame_count, 3), "object centers")
    _validate_array_shape(track["pinch_centers_base_m"], (frame_count, 3), "pinch centers")
    if not np.isfinite(timestamps_s).all() or np.any(np.diff(timestamps_s) <= 0.0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if not np.allclose(timestamps_s, track["timestamps_s"], atol=1e-6, rtol=0.0):
        raise ValueError("input and pickup-track timestamps do not match")

    pinch_camera_m = np.mean(joints_camera_m[:, [THUMB_TIP, INDEX_TIP]], axis=1)
    wrist_camera_m = joints_camera_m[:, WRIST]
    pinch_base_m = _transform_points(base_camera, pinch_camera_m)
    wrist_base_m = _transform_points(base_camera, wrist_camera_m)
    closing_camera = joints_camera_m[:, INDEX_TIP] - joints_camera_m[:, THUMB_TIP]
    closing_norm = np.linalg.norm(closing_camera, axis=1)
    closing_axis_base = np.full((frame_count, 3), np.nan, dtype=float)
    nonzero = closing_norm > 1e-8
    closing_axis_base[nonzero] = _transform_vectors(
        base_camera, closing_camera[nonzero] / closing_norm[nonzero, None]
    )
    pinch_confidence = np.minimum(confidence[:, THUMB_TIP], confidence[:, INDEX_TIP])
    pinch_valid = (
        (pinch_confidence >= args.min_joint_confidence)
        & np.isfinite(pinch_base_m).all(axis=1)
        & nonzero
    )
    wrist_valid = (confidence[:, WRIST] >= args.min_joint_confidence) & np.isfinite(
        wrist_base_m
    ).all(axis=1)

    intervals, diagnostics = _infer_stage_intervals(
        pickup, track, joints_camera_m, confidence, args
    )
    interval_by_name = {stage.name: stage for stage in intervals}
    grasp_translation = demonstrated_grasp[:3, 3].copy()

    def require_simplified(name: str, start: int, end: int) -> SimplifiedPath:
        path = _simplify_path(
            pinch_base_m,
            timestamps_s,
            pinch_valid,
            start,
            end,
            args,
        )
        if path is None:
            raise ValueError(f"{name} has fewer than two reliable trajectory samples")
        return path

    approach = interval_by_name["APPROACH"]
    preshape = interval_by_name["PRESHAPE"]
    lift = interval_by_name["LIFT"]
    place = interval_by_name["PLACE"]
    release = interval_by_name["RELEASE"]
    retreat = interval_by_name["RETREAT"]
    assert approach.start is not None and approach.end is not None
    assert preshape.start is not None and preshape.end is not None
    assert lift.start is not None and lift.end is not None
    assert place.start is not None and place.end is not None
    assert release.start is not None and release.end is not None
    assert retreat.start is not None and retreat.end is not None

    compiled_paths: dict[str, AnchoredPath] = {}
    approach_source = (approach.start, approach.end + 1)
    approach_path = require_simplified("APPROACH", *approach_source)
    compiled_paths["APPROACH"] = _anchor_path(
        "APPROACH",
        approach_path,
        anchor=AnchorFrame.GRASP_TRANSLATION,
        demonstrated_anchor_base_m=grasp_translation,
        source_interval=approach_source,
    )

    preshape_source = (preshape.start, preshape.end)
    preshape_path = require_simplified("PRESHAPE", *preshape_source)
    preshape_path = _force_path_boundary(
        preshape_path,
        side="start",
        frame=approach.end,
        timestamp_s=float(timestamps_s[approach.end]),
        position_base_m=approach_path.positions_base_m[-1],
    )
    preshape_path = _force_path_boundary(
        preshape_path,
        side="end",
        frame=preshape.end,
        timestamp_s=float(timestamps_s[preshape.end]),
        position_base_m=grasp_translation,
    )
    compiled_paths["PRESHAPE"] = _anchor_path(
        "PRESHAPE",
        preshape_path,
        anchor=AnchorFrame.GRASP_TRANSLATION,
        demonstrated_anchor_base_m=grasp_translation,
        source_interval=preshape_source,
        boundary_constraints=(
            "start_equals_approach_terminal",
            "terminal_equals_demonstrated_grasp_translation",
        ),
    )

    lift_source = (lift.start, lift.end + 1)
    lift_path = require_simplified("LIFT", *lift_source)
    lift_path = _force_path_boundary(
        lift_path,
        side="start",
        frame=lift.start,
        timestamp_s=float(timestamps_s[lift.start]),
        position_base_m=grasp_translation,
    )
    compiled_paths["LIFT"] = _anchor_path(
        "LIFT",
        lift_path,
        anchor=AnchorFrame.GRASP_TRANSLATION,
        demonstrated_anchor_base_m=grasp_translation,
        source_interval=lift_source,
        boundary_constraints=("initial_offset_is_zero",),
    )

    previous_terminal = lift_path.positions_base_m[-1]
    transport = interval_by_name["TRANSPORT"]
    if transport.enabled:
        assert transport.start is not None and transport.end is not None
        transport_source = (transport.start, transport.end + 1)
        transport_path = require_simplified("TRANSPORT", *transport_source)
        transport_path = _force_path_boundary(
            transport_path,
            side="start",
            frame=transport.start,
            timestamp_s=float(timestamps_s[transport.start]),
            position_base_m=previous_terminal,
        )
        compiled_paths["TRANSPORT"] = _anchor_path(
            "TRANSPORT",
            transport_path,
            anchor=AnchorFrame.RUNTIME_STAGE_START,
            demonstrated_anchor_base_m=previous_terminal,
            source_interval=transport_source,
            boundary_constraints=("initial_offset_is_zero",),
        )
        previous_terminal = transport_path.positions_base_m[-1]

    place_source = (place.start, place.end + 1)
    place_path = require_simplified("PLACE", *place_source)
    place_path = _force_path_boundary(
        place_path,
        side="start",
        frame=place.start,
        timestamp_s=float(timestamps_s[place.start]),
        position_base_m=previous_terminal,
    )
    release_pose = pinch_base_m[release.start].copy()
    place_path = _force_path_boundary(
        place_path,
        side="end",
        frame=release.start,
        timestamp_s=float(timestamps_s[release.start]),
        position_base_m=release_pose,
    )
    compiled_paths["PLACE"] = _anchor_path(
        "PLACE",
        place_path,
        anchor=AnchorFrame.RUNTIME_STAGE_START,
        demonstrated_anchor_base_m=previous_terminal,
        source_interval=place_source,
        boundary_constraints=(
            "initial_offset_is_zero",
            "terminal_equals_release_pose",
        ),
    )

    retreat_source = (release.start, retreat.end)
    retreat_path = require_simplified("RETREAT", *retreat_source)
    retreat_path = _force_path_boundary(
        retreat_path,
        side="start",
        frame=release.start,
        timestamp_s=float(timestamps_s[release.start]),
        position_base_m=release_pose,
    )
    compiled_paths["RETREAT"] = _anchor_path(
        "RETREAT",
        retreat_path,
        anchor=AnchorFrame.RELEASE_POSE,
        demonstrated_anchor_base_m=release_pose,
        source_interval=retreat_source,
        boundary_constraints=(
            "initial_offset_is_zero",
            "source_begins_at_release_start_before_retreat_stage_transition",
        ),
    )

    support_height_base_m = float(pickup["quality"]["local_support_height_base_m"])
    try:
        path_validation = _validate_compiled_paths(
            compiled_paths,
            support_height_base_m=support_height_base_m,
            min_support_clearance_m=args.min_support_clearance_m,
            max_continuity_gap_m=args.max_stage_continuity_gap_m,
            min_retreat_lift_m=args.min_retreat_lift_m,
            max_prelift_horizontal_m=args.max_prelift_horizontal_m,
        )
        path_validation.update(
            {
                "valid": True,
                "role": "diagnostic_only",
                "consumed_for_execution": False,
            }
        )
    except ValueError as exc:
        # Human free-space motion is useful evidence for phase interpretation,
        # but v3 never replays it. A diagnostic path defect therefore cannot
        # invalidate an otherwise usable task intent and selected grasp relation.
        path_validation = {
            "valid": False,
            "role": "diagnostic_only",
            "consumed_for_execution": False,
            "diagnostic_error": str(exc),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.npz"
    trajectory_arrays: dict[str, np.ndarray] = {
        "timestamps_s": timestamps_s,
        "object_centers_base_m": np.asarray(
            track["object_centers_base_m"], dtype=float
        ),
        "pinch_centers_base_m": pinch_base_m,
        "pinch_confidence": pinch_confidence,
        "pinch_valid": pinch_valid,
        "wrist_centers_base_m": wrist_base_m,
        "wrist_confidence": confidence[:, WRIST],
        "wrist_valid": wrist_valid,
        "closing_axes_base": closing_axis_base,
        "stage_names": np.asarray([stage.name for stage in intervals]),
        "stage_frame_intervals": np.asarray(
            [
                [
                    stage.start if stage.start is not None else -1,
                    stage.end if stage.end is not None else -1,
                ]
                for stage in intervals
            ],
            dtype=np.int64,
        ),
        "stage_enabled": np.asarray([stage.enabled for stage in intervals], dtype=bool),
    }
    for name, path in compiled_paths.items():
        prefix = name.lower()
        trajectory_arrays[f"{prefix}_frames"] = path.frames
        trajectory_arrays[f"{prefix}_position_offsets_base_m"] = (
            path.position_offsets_base_m
        )
    np.savez_compressed(trajectory_path, **trajectory_arrays)

    preview_path = output_dir / "preview.png"
    _write_static_preview(
        preview_path,
        rgb_paths,
        keypoints_2d,
        pinch_valid,
        intervals,
        compiled_paths,
        int(pickup["lift_onset_frame"]),
        diagnostics,
    )
    video_path: Path | None = None
    preview_fps: float | None = None
    if not args.skip_video:
        video_path = output_dir / "preview.mp4"
        preview_fps = _write_video_preview(
            video_path,
            rgb_paths,
            keypoints_2d,
            pinch_valid,
            timestamps_s,
            intervals,
            compiled_paths,
            args.video_fps,
        )

    contact_object_center = np.asarray(
        pickup["prelift_contact_object_center_base_m"], dtype=float
    )
    grasp_offset = demonstrated_grasp[:3, 3] - contact_object_center
    grasp_rotation = demonstrated_grasp[:3, :3]
    planar_closing_axis = grasp_rotation[:, 0].copy()
    planar_closing_axis[2] = 0.0
    planar_closing_norm = float(np.linalg.norm(planar_closing_axis[:2]))
    planar_closing_axis_diagnostic = (
        None
        if planar_closing_norm <= 1e-6
        else (planar_closing_axis / planar_closing_norm).tolist()
    )
    task_stages = tuple(
        _task_skill_stage(stage, timestamps_s, compiled_paths.get(stage.name))
        for stage in intervals
        if stage.enabled and stage.name != "OBSERVE" and stage.frame_count > 0
    )
    blocking_conditions = tuple(
        dict.fromkeys(
            [
                "hande_command_frame_mapping_unchecked",
                "ik_unchecked",
                "collision_unchecked",
            ]
        )
    )
    task_skill = TaskSkill(
        skill_id=f"{demo_dir.name}_prototype_v0",
        stages=task_stages,
        grasp_relation=ObjectGraspRelation.se3(
            T_object_gripper_tcp[:3, 3],
            T_object_gripper_tcp[:3, :3],
            approach_distance_m=selected_approach_distance_m,
            object_frame_id=str(selected_grasp["object_frame_id"]),
            gripper_name=SELECTED_GRASP_GRIPPER_NAME,
            jaw_opening_m=selected_pregrasp_width_m,
        ),
        status=SkillStatus.REVIEW_ONLY,
        execution_authorized=False,
        robot_execution_ready=False,
        blocking_conditions=blocking_conditions,
    )
    stage_contracts: list[dict[str, Any]] = []
    task_stage_by_name = {stage.name.value: stage for stage in task_skill.stages}
    for stage in intervals:
        compiled_path = compiled_paths.get(stage.name)
        canonical_stage = task_stage_by_name.get(stage.name)
        contract = _stage_contract(
            stage,
            compiled_path is not None,
            None if canonical_stage is None else canonical_stage.anchor,
        )
        if compiled_path is not None:
            contract["trajectory"] = _path_json(
                compiled_path, stage.name.lower()
            )
        stage_contracts.append(contract)

    report = {
        "schema": "demo2gripper.agentic_skill_compiler_report.v1",
        "mode": "prototype_phase_intent_plus_selected_object_se3_grasp",
        "status": "review_only",
        "execution_authorized": False,
        "robot_execution_ready": False,
        "description": (
            "Human motion supplies phase intent; the selected Hand-E candidate supplies "
            "the immutable object-relative GRASP endpoint. The runtime agent generates "
            "free-space stage targets, asks MoveIt to plan, and verifies each exit guard."
        ),
        "coordinate_contract": {
            "convention": "right-handed T_parent_child, column vectors, metres",
            "demonstration_path_frame": "FR3 base",
            "path_proxy": "human thumb/index pinch center",
            "spatial_transfer_model": "object_relative_se3",
            "object_rotation_transfer_supported": True,
            "required_live_input": "rigid T_base_object with matching object_frame_id",
        },
        "grasp_intent": {
            "contract": "selected_robot_specific_object_relative_se3",
            "frame_convention": SELECTED_GRASP_FRAME_CONVENTION,
            "object_frame_id": selected_grasp["object_frame_id"],
            "gripper_name": SELECTED_GRASP_GRIPPER_NAME,
            "intent_provenance": selected_grasp["intent_provenance"],
            "selection": selected_grasp["selection"],
            "T_object_gripper_tcp": T_object_gripper_tcp.tolist(),
            "pregrasp_width_m": selected_pregrasp_width_m,
            "approach_distance_m": selected_approach_distance_m,
            "demonstrated_object_center_base_m": contact_object_center.tolist(),
            "grasp_origin_offset_from_object_center_in_demo_base_axes_m": grasp_offset.tolist(),
            "demonstrated_grasp_origin_base_m": demonstrated_grasp[:3, 3].tolist(),
            "demonstrated_human_grasp_rotation_base": grasp_rotation.tolist(),
            "demonstrated_closing_axis_base_diagnostic_only": (
                planar_closing_axis_diagnostic
            ),
            "closing_axis_is_undirected": True,
            "closing_axis_period_rad": float(np.pi),
            "demonstrated_approach_axis_base_diagnostic_only": grasp_rotation[
                :, 2
            ].tolist(),
            "demonstrated_human_pinch_width_m": float(pickup["demonstrated_pinch_width_m"]),
            "human_pinch_is_not_robot_width": True,
            "recommended_review_opening_m": float(pickup["recommended_pregrasp_opening_m"]),
            "orientation_transferability": (
                "the complete selected grasp transfers through a matching live rigid "
                "T_base_object"
            ),
            "robot_specific_candidate_available": True,
            "robot_command_frame_mapping_available": False,
        },
        "runtime_control_contract": {
            "agent_responsibilities": [
                "select the active stage",
                "ground the selected object-frame SE(3) endpoint in the current scene",
                "plan a collision-free route to the selected local-minus-Z pregrasp",
                "generate lift, placement, and retreat targets online",
                "request a collision-checked motion plan",
                "verify the exit guard",
                "reobserve, replan, or recover after failure",
            ],
            "moveit_responsibilities": [
                "IK and reachability",
                "collision checking",
                "time-parameterized robot joint trajectory",
            ],
            "required_before_robot_execution": [
                "fresh accepted rigid T_base_object and selected Hand-E SE(3) grasp",
                "measured Hand-E command-frame-to-TCP transform",
                "Hand-E URDF/SRDF and collision geometry",
                "MoveIt planning-scene, IK, and collision validation",
                "grasp and placement verification signals",
            ],
        },
        "phase_graph": {
            "entry": "OBSERVE",
            "nominal_order": [stage.name for stage in intervals if stage.enabled],
            "replan_policy": (
                "hold on stale perception; reobserve and reground after object motion; "
                "never send human proxy coordinates directly to a robot controller"
            ),
            "stages": stage_contracts,
        },
        "task_skill": task_skill.to_dict(),
        "path_validation": path_validation,
        "compiler_diagnostics": diagnostics,
        "compiler_config": {
            "min_joint_confidence": args.min_joint_confidence,
            "hand_presence_confidence": args.hand_presence_confidence,
            "rdp_epsilon_m": args.rdp_epsilon_m,
            "min_waypoints": args.min_waypoints,
            "max_waypoints": args.max_waypoints,
            "smoothing_window_frames": args.smoothing_window_frames,
            "release_opening_delta_m": args.release_opening_delta_m,
            "release_retreat_distance_m": args.release_retreat_distance_m,
            "retreat_clearance_m": args.retreat_clearance_m,
            "transport_min_horizontal_m": args.transport_min_horizontal_m,
            "transport_height_fraction": args.transport_height_fraction,
            "approach_reliability_window_frames": (
                args.approach_reliability_window_frames
            ),
            "approach_reliability_fraction": args.approach_reliability_fraction,
            "min_retreat_lift_m": args.min_retreat_lift_m,
            "max_prelift_horizontal_m": args.max_prelift_horizontal_m,
            "min_support_clearance_m": args.min_support_clearance_m,
            "max_stage_continuity_gap_m": args.max_stage_continuity_gap_m,
        },
        "source_quality": {
            "pickup_demonstration_validated": pickup[
                "pickup_demonstration_validated"
            ],
            "fixed_object_replay_valid": pickup["fixed_object_replay_valid"],
            "fixed_object_replay_gate_failures": pickup.get(
                "fixed_object_replay_gate_failures", []
            ),
            "fixed_object_replay_assessment_role": "source_diagnostic_only",
            "robot_command_frame_available": pickup.get(
                "robot_command_frame_available", False
            ),
            "robot_grasp_width_feasible": pickup.get(
                "robot_grasp_width_feasible"
            ),
            "grasp_plane_width_validated": pickup.get(
                "grasp_plane_width_validated", False
            ),
            "blocking_conditions": list(blocking_conditions),
        },
        "artifacts": {
            "task_skill": "agentic_skill.json",
            "compiler_report": "compiler_report.json",
            "trajectory": trajectory_path.name,
            "trajectory_role": "diagnostic_only",
            "trajectory_consumed_for_execution": False,
            "preview": preview_path.name,
            "video": video_path.name if video_path is not None else None,
            "video_fps": preview_fps,
        },
        "source": {
            "demo_dir": str(demo_dir.resolve()),
            "pickup_grasp": str(pickup_path.resolve()),
            "pickup_track": str(track_path.resolve()),
            "selected_grasp": str(selected_grasp_path),
            "hand_joints_camera_m": str(joints_path.resolve()),
            "hand_joint_confidence": str(confidence_path.resolve()),
            "hand_keypoints_2d": str(keypoints_path.resolve()),
            "timestamps_s": str(timestamps_path.resolve()),
        },
        "limitations": [
            "Human pinch-center paths are retained only for phase audit and preview.",
            "It does not run IK, collision checking, motion planning, force checks, or control.",
            "Runtime grounding requires a fresh full rigid T_base_object, not a center anchor.",
            "The phase boundaries are heuristic and must be reviewed in preview.mp4.",
        ],
    }
    skill_path = output_dir / "agentic_skill.json"
    skill_path.write_text(task_skill.to_json(indent=2) + "\n", encoding="utf-8")
    report_path = output_dir / "compiler_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return skill_path, report_path, trajectory_path, preview_path, video_path


def main() -> None:
    args = parse_args()
    skill_path, report_path, trajectory_path, preview_path, video_path = compile_skill(args)
    print(f"wrote {skill_path}")
    print(f"wrote {report_path}")
    print(f"wrote {trajectory_path}")
    print(f"wrote {preview_path}")
    if video_path is not None:
        print(f"wrote {video_path}")
    print("status: review_only; execution_authorized=false")


if __name__ == "__main__":
    main()
