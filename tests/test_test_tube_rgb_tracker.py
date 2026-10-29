from __future__ import annotations

import math

import numpy as np
import pytest

from demo2gripper.test_tube_rgb_tracker import (
    ImageLineCandidate,
    TubeAxisCandidateTracker,
    TubeAxisTrackerConfig,
    extract_hough_line_candidates,
)
from demo2gripper.test_tube_visual import VisualAngleGate, VisualAngleGateConfig


def _line(
    angle_deg: float,
    *,
    center: tuple[float, float] = (200.0, 100.0),
    length: float = 120.0,
) -> ImageLineCandidate:
    angle = math.radians(angle_deg)
    half_delta = 0.5 * length * np.array([math.cos(angle), math.sin(angle)])
    middle = np.asarray(center, dtype=float)
    start = middle - half_delta
    end = middle + half_delta
    return ImageLineCandidate(tuple(start), tuple(end))


def test_line_candidate_validates_its_geometry() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ImageLineCandidate((1.0, 2.0), (1.0, 2.0))
    with pytest.raises(ValueError, match="edge_support"):
        ImageLineCandidate((0.0, 0.0), (10.0, 0.0), edge_support=1.1)


def test_tracker_requires_reviewed_seed_and_tracks_continuous_line() -> None:
    tracker = TubeAxisCandidateTracker()
    missing_seed = tracker.update(timestamp_s=0.0, candidates=[_line(2.0)])
    assert missing_seed.decision == "RESEED_REQUIRED"
    assert not missing_seed.observation.valid

    seed = tracker.seed(timestamp_s=1.0, line=_line(0.0))
    assert seed.valid
    result = tracker.update(timestamp_s=1.1, candidates=[_line(2.0, center=(204.0, 103.0))])
    assert result.decision == "TRACKED"
    assert result.observation.valid
    assert result.observation.axis_angle_deg == pytest.approx(2.0)


def test_distinct_equal_hypotheses_fail_closed() -> None:
    tracker = TubeAxisCandidateTracker()
    tracker.seed(timestamp_s=0.0, line=_line(0.0))
    result = tracker.update(
        timestamp_s=0.1,
        candidates=[
            _line(0.0, center=(200.0, 70.0)),
            _line(0.0, center=(200.0, 130.0)),
        ],
    )
    assert result.decision == "HOLD_AND_REOBSERVE"
    assert "indistinguishable" in result.reason
    assert not result.observation.valid


def test_parallel_segments_on_same_tube_are_not_false_ambiguity() -> None:
    tracker = TubeAxisCandidateTracker()
    tracker.seed(timestamp_s=0.0, line=_line(0.0))
    result = tracker.update(
        timestamp_s=0.1,
        candidates=[
            _line(1.0, center=(200.0, 94.0), length=130.0),
            _line(1.5, center=(200.0, 108.0), length=118.0),
        ],
    )
    assert result.decision == "TRACKED"
    assert result.equivalent_candidate_count == 2


def test_tracker_does_not_reacquire_after_long_gap() -> None:
    tracker = TubeAxisCandidateTracker(TubeAxisTrackerConfig(max_tracking_gap_s=0.2))
    tracker.seed(timestamp_s=0.0, line=_line(0.0))
    first = tracker.update(timestamp_s=0.1, candidates=[])
    second = tracker.update(timestamp_s=0.21, candidates=[_line(0.0)])
    assert first.decision == "HOLD_AND_REOBSERVE"
    assert second.decision == "RESEED_REQUIRED"
    assert not tracker.seeded


def test_roi_rejects_candidate_outside_tracking_window() -> None:
    tracker = TubeAxisCandidateTracker()
    tracker.seed(timestamp_s=0.0, line=_line(0.0))
    result = tracker.update(
        timestamp_s=0.1,
        candidates=[_line(0.0, center=(300.0, 100.0))],
        roi_uvxy=(100.0, 50.0, 250.0, 150.0),
    )
    assert result.decision == "HOLD_AND_REOBSERVE"


def test_tracker_observation_drives_existing_consecutive_angle_gate() -> None:
    tracker = TubeAxisCandidateTracker()
    seed = tracker.seed(timestamp_s=0.0, line=_line(146.0))
    gate = VisualAngleGate(
        VisualAngleGateConfig(
            target_image_axis_deg=135.5,
            tolerance_deg=5.0,
            min_consecutive_frames=4,
        )
    )
    gate.update(timestamp_s=0.0, observation=seed)
    decisions = []
    for index, angle in enumerate((140.0, 138.0, 136.0, 135.0), start=1):
        tracked = tracker.update(timestamp_s=index * 0.1, candidates=[_line(angle)])
        assert tracked.decision == "TRACKED"
        decisions.append(
            gate.update(timestamp_s=index * 0.1, observation=tracked.observation).decision
        )
    assert decisions == [
        "CONTINUE_BOUNDED_PIVOT",
        "CONTINUE_BOUNDED_PIVOT",
        "CONTINUE_BOUNDED_PIVOT",
        "RECLAMP",
    ]


def test_cpu_hough_extractor_returns_line_in_roi() -> None:
    cv2 = pytest.importorskip("cv2")
    image = np.zeros((180, 260, 3), dtype=np.uint8)
    cv2.line(image, (60, 140), (200, 40), (255, 255, 255), 3)
    candidates = extract_hough_line_candidates(
        image,
        roi_uvxy=(40, 20, 220, 160),
        hough_threshold=15,
        min_line_length_px=40,
    )
    assert candidates
    assert max(candidate.length_px for candidate in candidates) > 100.0
