"""Rank object-frame grasp candidates against a demonstrated intent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import canonicalize_continuous_symmetry_pose
from .models import (
    GraspCandidates,
    GraspIntent,
    JsonSerializable,
    RankedCandidate,
)


@dataclass(frozen=True)
class SelectionConfig(JsonSerializable):
    min_gripper_width_m: float = 0.0
    max_gripper_width_m: float = 0.085
    max_center_distance_m: float = np.inf
    max_closing_axis_error_rad: float = np.pi
    max_approach_axis_error_rad: float = np.pi
    center_scale_m: float = 0.05
    center_weight: float = 1.0
    closing_axis_weight: float = 0.5
    approach_axis_weight: float = 0.5
    model_score_weight: float = 0.25


def _axis_angle(first: np.ndarray, second: np.ndarray, *, unsigned: bool) -> float:
    cosine = float(np.dot(first, second))
    if unsigned:
        cosine = abs(cosine)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def rank_grasp_candidates(
    intent: GraspIntent,
    candidates: GraspCandidates,
    config: SelectionConfig = SelectionConfig(),
) -> list[RankedCandidate]:
    """Return feasible candidates ordered from best to worst."""

    width_feasible = (
        (candidates.widths_m >= config.min_gripper_width_m)
        & (candidates.widths_m <= config.max_gripper_width_m)
    )
    intent_pose = intent.T_object_gripper_tcp
    comparison_poses = candidates.poses
    if intent.continuous_symmetry_axis is not None:
        intent_pose = canonicalize_continuous_symmetry_pose(
            intent_pose, intent.continuous_symmetry_axis
        )
        comparison_poses = np.stack(
            [
                canonicalize_continuous_symmetry_pose(
                    pose, intent.continuous_symmetry_axis
                )
                for pose in candidates.poses
            ]
        )
    center_errors = np.linalg.norm(
        comparison_poses[:, :3, 3] - intent_pose[:3, 3], axis=1
    )
    closing_errors = np.array(
        [
            _axis_angle(pose[:3, 0], intent_pose[:3, 0], unsigned=True)
            for pose in comparison_poses
        ]
    )
    approach_errors = np.array(
        [
            _axis_angle(pose[:3, 2], intent_pose[:3, 2], unsigned=False)
            for pose in comparison_poses
        ]
    )
    feasible = (
        width_feasible
        & (center_errors <= config.max_center_distance_m)
        & (closing_errors <= config.max_closing_axis_error_rad)
        & (approach_errors <= config.max_approach_axis_error_rad)
    )
    indices = np.flatnonzero(feasible)
    if len(indices) == 0:
        raise ValueError("no candidates satisfy the width and pose-error limits")

    feasible_scores = np.asarray(candidates.scores[indices], dtype=float)
    score_span = float(np.ptp(feasible_scores))
    if score_span > 0.0:
        normalized_scores = (feasible_scores - np.min(feasible_scores)) / score_span
    else:
        normalized_scores = np.zeros_like(feasible_scores)

    ranked: list[RankedCandidate] = []
    for candidate_index, normalized_score in zip(indices, normalized_scores, strict=True):
        pose = candidates.poses[candidate_index]
        center_error = float(center_errors[candidate_index])
        closing_error = float(closing_errors[candidate_index])
        approach_error = float(approach_errors[candidate_index])
        cost = (
            config.center_weight * center_error / config.center_scale_m
            + config.closing_axis_weight * closing_error / np.pi
            + config.approach_axis_weight * approach_error / np.pi
            - config.model_score_weight * float(normalized_score)
        )
        ranked.append(
            RankedCandidate(
                index=int(candidate_index),
                T_object_gripper_tcp=np.asarray(pose, dtype=float),
                width_m=float(candidates.widths_m[candidate_index]),
                model_score=float(candidates.scores[candidate_index]),
                center_error_m=center_error,
                closing_axis_error_rad=closing_error,
                approach_axis_error_rad=approach_error,
                ranking_cost=float(cost),
            )
        )

    return sorted(ranked, key=lambda candidate: (candidate.ranking_cost, candidate.index))


def select_grasp_candidate(
    intent: GraspIntent,
    candidates: GraspCandidates,
    config: SelectionConfig = SelectionConfig(),
) -> RankedCandidate:
    return rank_grasp_candidates(intent, candidates, config)[0]
