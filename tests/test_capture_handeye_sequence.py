from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.capture_handeye_sequence import (
    TimedPose,
    moved_enough,
    pose_novelty,
    pose_window_is_stationary,
    transform_difference,
    trim_pose_window,
)


def _pose(
    *, translation: tuple[float, float, float] = (0.0, 0.0, 0.0), rz_deg: float = 0.0
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3], _ = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(rz_deg)]))
    transform[:3, 3] = translation
    return transform


def test_transform_difference_and_rearm_use_translation_or_rotation() -> None:
    origin = _pose()
    translated = _pose(translation=(0.020, 0.0, 0.0))
    rotated = _pose(rz_deg=8.0)

    difference = transform_difference(origin, translated)
    assert difference.translation_m == 0.020
    assert difference.rotation_deg == 0.0
    assert moved_enough(
        origin,
        translated,
        minimum_translation_m=0.015,
        minimum_rotation_deg=6.0,
    )[0]
    assert moved_enough(
        origin,
        rotated,
        minimum_translation_m=0.015,
        minimum_rotation_deg=6.0,
    )[0]
    assert not moved_enough(
        origin,
        _pose(translation=(0.005, 0.0, 0.0), rz_deg=2.0),
        minimum_translation_m=0.015,
        minimum_rotation_deg=6.0,
    )[0]


def test_stationary_window_requires_duration_continuity_and_small_spread() -> None:
    stable = deque(
        TimedPose(
            host_monotonic_ns=int(index * 0.2e9),
            transform=_pose(translation=(index * 0.0001, 0.0, 0.0), rz_deg=index * 0.02),
        )
        for index in range(5)
    )
    stationary, spread = pose_window_is_stationary(
        stable,
        minimum_duration_s=0.75,
        minimum_observations=5,
        maximum_gap_s=0.3,
        maximum_translation_m=0.002,
        maximum_rotation_deg=0.5,
    )
    assert stationary
    assert spread is not None
    assert spread.translation_m < 0.001
    assert spread.rotation_deg < 0.5

    unstable = deque(stable)
    unstable[-1] = TimedPose(unstable[-1].host_monotonic_ns, _pose(rz_deg=2.0))
    assert not pose_window_is_stationary(
        unstable,
        minimum_duration_s=0.75,
        minimum_observations=5,
        maximum_gap_s=0.3,
        maximum_translation_m=0.002,
        maximum_rotation_deg=0.5,
    )[0]


def test_trim_pose_window_keeps_boundary_observation() -> None:
    history = deque(TimedPose(index * 100_000_000, _pose()) for index in range(20))
    trim_pose_window(history, 1_900_000_000, 0.75)
    assert history[0].host_monotonic_ns == 1_100_000_000
    assert history[1].host_monotonic_ns == 1_200_000_000


def test_pose_novelty_rejects_only_when_both_differences_are_small() -> None:
    references = [_pose(), _pose(translation=(0.2, 0.0, 0.0), rz_deg=20.0)]
    duplicate = pose_novelty(
        _pose(translation=(0.005, 0.0, 0.0), rz_deg=2.0),
        references,
        minimum_translation_m=0.015,
        minimum_rotation_deg=6.0,
    )
    assert not duplicate.novel
    assert duplicate.nearest_index == 0

    translated = pose_novelty(
        _pose(translation=(0.030, 0.0, 0.0), rz_deg=1.0),
        references,
        minimum_translation_m=0.015,
        minimum_rotation_deg=6.0,
    )
    assert translated.novel

    rotated = pose_novelty(
        _pose(translation=(0.002, 0.0, 0.0), rz_deg=10.0),
        references,
        minimum_translation_m=0.015,
        minimum_rotation_deg=6.0,
    )
    assert rotated.novel

    # A non-duplicate can have a smaller normalized distance than a reference
    # that is just inside both limits; every reference must still be checked.
    tricky_references = [
        _pose(translation=(0.011, 0.0, 0.0), rz_deg=0.0),
        _pose(translation=(0.008, 0.0, 0.0), rz_deg=4.0),
    ]
    tricky = pose_novelty(
        _pose(),
        tricky_references,
        minimum_translation_m=0.010,
        minimum_rotation_deg=5.0,
    )
    assert not tricky.novel
    assert tricky.nearest_index == 1
