#!/usr/bin/env python3
"""Select a CAD-valid parallel-jaw grasp from a demonstrated power grip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


THUMB_JOINTS = (2, 3, 4)
OPPOSITION_JOINTS = (6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--t-camera-object", required=True, type=Path)
    parser.add_argument("--object-points", required=True, type=Path)
    parser.add_argument("--keypoints-2d", required=True, type=Path)
    parser.add_argument("--scores-2d", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--intent-output", required=True, type=Path)
    parser.add_argument("--selected-output", required=True, type=Path)
    parser.add_argument("--min-keypoint-score", type=float, default=0.4)
    parser.add_argument("--max-association-error-px", type=float, default=10.0)
    parser.add_argument("--max-center-deviation-m", type=float, default=0.02)
    parser.add_argument("--min-human-width-m", type=float, default=0.01)
    parser.add_argument("--max-human-width-m", type=float, default=0.065)
    parser.add_argument("--min-stable-frames", type=int, default=3)
    parser.add_argument("--contact-line-radius-m", type=float, default=0.005)
    parser.add_argument("--min-contact-width-m", type=float, default=0.01)
    parser.add_argument("--max-contact-width-m", type=float, default=0.08)
    parser.add_argument("--max-center-distance-m", type=float, default=0.06)
    parser.add_argument("--approach-distance-m", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def project(
    points_object: np.ndarray, transform_camera_object: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points_camera = (
        points_object @ transform_camera_object[:3, :3].T
        + transform_camera_object[:3, 3]
    )
    homogeneous = points_camera @ intrinsics.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, points_camera[:, 2]


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError("cannot normalize a zero-length direction")
    return np.asarray(vector, dtype=float) / norm


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    if not len(transitions):
        return 0, 0
    starts, ends = transitions[::2], transitions[1::2]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def robust_mean(samples: np.ndarray) -> np.ndarray:
    center = np.median(samples, axis=0)
    residual = np.linalg.norm(samples - center, axis=1)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    keep = residual <= median + 3.0 * max(mad, 1e-9)
    return np.mean(samples[keep], axis=0)


def associate_group(
    joint_ids: tuple[int, ...],
    keypoints: np.ndarray,
    scores: np.ndarray,
    tree: cKDTree,
    visible_object_points: np.ndarray,
    *,
    min_score: float,
    max_error_px: float,
    min_matches: int,
) -> tuple[np.ndarray, float] | None:
    ids = np.asarray(joint_ids, dtype=int)
    ids = ids[scores[ids] >= min_score]
    if len(ids) < min_matches:
        return None
    distances, indices = tree.query(keypoints[ids], k=1)
    keep = distances <= max_error_px
    if int(np.count_nonzero(keep)) < min_matches:
        return None
    associated = visible_object_points[np.asarray(indices)[keep]]
    return np.median(associated, axis=0), float(np.median(distances[keep]))


def contact_line_intersections(
    center: np.ndarray,
    closing_axis: np.ndarray,
    object_points: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, float] | None:
    delta = object_points - center
    signed = delta @ closing_axis
    radial = np.linalg.norm(delta - signed[:, None] * closing_axis, axis=1)
    in_tube = radial <= radius_m
    negative = np.flatnonzero(in_tube & (signed < 0.0))
    positive = np.flatnonzero(in_tube & (signed > 0.0))
    if not len(negative) or not len(positive):
        return None
    first = negative[np.argmax(signed[negative])]
    second = positive[np.argmin(signed[positive])]
    contacts = object_points[[first, second]]
    width = float(signed[second] - signed[first])
    return contacts, width


def main() -> int:
    args = parse_args()
    transforms = np.asarray(np.load(args.t_camera_object, allow_pickle=False), dtype=float)
    object_points = np.asarray(np.load(args.object_points, allow_pickle=False), dtype=float)
    keypoints = np.asarray(np.load(args.keypoints_2d, allow_pickle=False), dtype=float)
    scores = np.asarray(np.load(args.scores_2d, allow_pickle=False), dtype=float)
    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=float)
    with np.load(args.candidates, allow_pickle=False) as candidate_data:
        candidate_poses = np.asarray(candidate_data["poses"], dtype=float)
        candidate_scores = np.asarray(candidate_data["scores"], dtype=float)
        candidate_widths = np.asarray(candidate_data["widths_m"], dtype=float)

    frame_count = len(transforms)
    if keypoints.shape != (frame_count, 21, 2) or scores.shape != (frame_count, 21):
        raise ValueError("RTMPose arrays must have shapes (F,21,2) and (F,21)")
    if intrinsics.shape != (3, 3) or object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("intrinsics or object point cloud has an invalid shape")

    centers = np.full((frame_count, 3), np.nan, dtype=float)
    axes = np.full((frame_count, 3), np.nan, dtype=float)
    widths = np.full(frame_count, np.nan, dtype=float)
    reprojection_errors = np.full(frame_count, np.nan, dtype=float)

    for frame_index in range(frame_count):
        pixels, depths = project(object_points, transforms[frame_index], intrinsics)
        visible = np.isfinite(pixels).all(axis=1) & (depths > 0.0)
        visible_pixels = pixels[visible]
        visible_points = object_points[visible]
        tree = cKDTree(visible_pixels)
        thumb = associate_group(
            THUMB_JOINTS,
            keypoints[frame_index],
            scores[frame_index],
            tree,
            visible_points,
            min_score=args.min_keypoint_score,
            max_error_px=args.max_association_error_px,
            min_matches=1,
        )
        fingers = associate_group(
            OPPOSITION_JOINTS,
            keypoints[frame_index],
            scores[frame_index],
            tree,
            visible_points,
            min_score=args.min_keypoint_score,
            max_error_px=args.max_association_error_px,
            min_matches=3,
        )
        if thumb is None or fingers is None:
            continue
        thumb_point, thumb_error = thumb
        finger_point, finger_error = fingers
        opposition = finger_point - thumb_point
        width = float(np.linalg.norm(opposition))
        if width <= 1e-9:
            continue
        centers[frame_index] = 0.5 * (thumb_point + finger_point)
        axes[frame_index] = opposition / width
        widths[frame_index] = width
        reprojection_errors[frame_index] = 0.5 * (thumb_error + finger_error)

    preliminary = (
        np.isfinite(centers).all(axis=1)
        & (widths >= args.min_human_width_m)
        & (widths <= args.max_human_width_m)
    )
    if int(np.count_nonzero(preliminary)) < args.min_stable_frames:
        raise ValueError("too few frames contain a measurable power-grip opposition")
    center_reference = robust_mean(centers[preliminary])
    axis_reference = normalized(np.median(axes[preliminary], axis=0))
    aligned_axes = axes.copy()
    flip = np.einsum("ij,j->i", aligned_axes[preliminary], axis_reference) < 0.0
    preliminary_indices = np.flatnonzero(preliminary)
    aligned_axes[preliminary_indices[flip]] *= -1.0
    axis_reference = normalized(np.mean(aligned_axes[preliminary], axis=0))
    center_deviation = np.linalg.norm(centers - center_reference, axis=1)
    stable = preliminary & (center_deviation <= args.max_center_deviation_m)
    start, end = longest_true_run(stable)
    if end - start < args.min_stable_frames:
        raise ValueError(
            f"no stable power-grip interval with at least {args.min_stable_frames} frames"
        )

    interval = np.arange(start, end)
    grip_center = robust_mean(centers[interval])
    human_axis = normalized(np.mean(aligned_axes[interval], axis=0))
    human_width = float(np.median(widths[interval]))
    reprojection_error = float(np.median(reprojection_errors[interval]))

    score_span = float(np.ptp(candidate_scores))
    normalized_scores = (
        (candidate_scores - np.min(candidate_scores)) / score_span
        if score_span > 0.0
        else np.zeros_like(candidate_scores)
    )
    ranked: list[dict[str, object]] = []
    for index, (pose, model_score, pregrasp_width) in enumerate(
        zip(candidate_poses, candidate_scores, candidate_widths, strict=True)
    ):
        contact = contact_line_intersections(
            pose[:3, 3], pose[:3, 0], object_points, args.contact_line_radius_m
        )
        if contact is None:
            continue
        contact_points, contact_width = contact
        center_error = float(np.linalg.norm(pose[:3, 3] - grip_center))
        if not (
            args.min_contact_width_m <= contact_width <= args.max_contact_width_m
            and center_error <= args.max_center_distance_m
        ):
            continue
        closing_error = float(
            np.arccos(np.clip(abs(np.dot(pose[:3, 0], human_axis)), -1.0, 1.0))
        )
        cost = (
            center_error / 0.05
            + 0.25 * closing_error / np.pi
            - 0.20 * float(normalized_scores[index])
        )
        ranked.append(
            {
                "index": index,
                "T_object_gripper_tcp": pose.tolist(),
                "width_m": float(pregrasp_width),
                "contact_width_m": contact_width,
                "contact_points_object_m": contact_points.tolist(),
                "model_score": float(model_score),
                "center_error_m": center_error,
                "closing_axis_error_rad": closing_error,
                "approach_axis_error_rad": None,
                "ranking_cost": cost,
            }
        )
    if not ranked:
        raise ValueError("no GraspGenX candidate has valid bilateral CAD contacts")
    ranked.sort(key=lambda item: (float(item["ranking_cost"]), int(item["index"])))
    selected = ranked[0]

    provenance = {
        "T_camera_object": str(args.t_camera_object.resolve()),
        "object_points": str(args.object_points.resolve()),
        "hand_keypoints_2d": str(args.keypoints_2d.resolve()),
        "hand_scores_2d": str(args.scores_2d.resolve()),
        "candidates": str(args.candidates.resolve()),
    }
    intent = {
        "T_object_gripper_tcp": selected["T_object_gripper_tcp"],
        "width_m": selected["contact_width_m"],
        "contact_points_object_m": selected["contact_points_object_m"],
        "stable_interval": [start, end],
        "support_frames": end - start,
        "continuous_symmetry_axis": None,
        "intent_type": "power",
        "evidence_kind": "rtmpose2d_cad_reprojection",
        "approach_source": "graspgenx_candidate",
        "contact_source": "cad_closing_line",
        "candidate_index": selected["index"],
        "human_grip_center_object_m": grip_center.tolist(),
        "human_opposition_axis_object": human_axis.tolist(),
        "median_visible_group_width_m": human_width,
        "median_reprojection_error_px": reprojection_error,
        "frame_count": frame_count,
        "provenance": provenance,
    }
    selected_output = {
        "schema_version": 1,
        "frame_convention": "T_object_gripper_tcp; +X closing, +Z approach",
        "T_object_gripper_tcp": selected["T_object_gripper_tcp"],
        "pregrasp_width_m": selected["width_m"],
        "close_target_width_m": 0.0,
        "demonstrated_contact_width_m": selected["contact_width_m"],
        "approach_distance_m": args.approach_distance_m,
        "candidate_index": selected["index"],
        "model_score": selected["model_score"],
        "center_error_m": selected["center_error_m"],
        "closing_axis_error_rad": selected["closing_axis_error_rad"],
        "approach_axis_error_rad": None,
        "ranking_cost": selected["ranking_cost"],
        "selection_mode": "power_grip_cad_contact",
        "ranked_candidates": ranked[: args.top_k],
        "provenance": provenance,
    }
    args.intent_output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_output.parent.mkdir(parents=True, exist_ok=True)
    args.intent_output.write_text(json.dumps(intent, indent=2) + "\n", encoding="utf-8")
    args.selected_output.write_text(
        json.dumps(selected_output, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"power-grip interval {start}:{end}; selected GraspGenX candidate "
        f"{selected['index']} with {float(selected['contact_width_m']) * 1000.0:.1f} mm "
        f"CAD contact width"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
