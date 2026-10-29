"""Deterministic promotion checks for the test-tube front-RGB axis gate.

This module evaluates already-recorded tracker evidence.  It does not import a
camera SDK, ROS, a gripper driver, or any robot interface.  The caller remains
responsible for replaying recorded RGB through ``TubeAxisCandidateTracker`` and
for supplying independent human labels.

The promotion boundary is deliberately narrow: three robot-held recordings
with distinct immutable session and full-media identities must all satisfy the
same fixed criteria, and both an occlusion case and a distinct distractor-axis
case must fail closed in every trial.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

FAIL_CLOSED_TRACKER_DECISIONS = frozenset({"HOLD_AND_REOBSERVE", "RESEED_REQUIRED"})
REQUIRED_NEGATIVE_CASE_KINDS = frozenset({"partial_occlusion", "distractor_ambiguity"})


@dataclass(frozen=True)
class RGBTrackerPromotionCriteria:
    """Fixed evidence thresholds for promoting the robot-held RGB gate."""

    successful_trials_required: int = 3
    tracked_fraction_while_labelled_visible_min: float = 0.95
    reviewed_axis_mae_deg_max: float = 3.0
    reviewed_axis_max_error_deg_max: float = 5.0
    reclamp_frame_error_max: int = 3
    false_reclamp_before_labelled_target_max: int = 0
    max_axis_label_stride_frames: int = 5
    min_reviewed_axis_labels: int = 5
    require_occlusion_fail_closed: bool = True
    require_ambiguity_fail_closed: bool = True

    def __post_init__(self) -> None:
        if self.successful_trials_required < 1:
            raise ValueError("successful_trials_required must be positive")
        if not 0.0 < self.tracked_fraction_while_labelled_visible_min <= 1.0:
            raise ValueError("tracked fraction threshold must lie in (0, 1]")
        for name in ("reviewed_axis_mae_deg_max", "reviewed_axis_max_error_deg_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.reviewed_axis_max_error_deg_max < self.reviewed_axis_mae_deg_max:
            raise ValueError("maximum error threshold cannot be below MAE threshold")
        for name in (
            "reclamp_frame_error_max",
            "false_reclamp_before_labelled_target_max",
            "max_axis_label_stride_frames",
            "min_reviewed_axis_labels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


CANONICAL_RGB_TRACKER_PROMOTION_CRITERIA = RGBTrackerPromotionCriteria()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ImmutableRecordingEvidence:
    """Annotation-independent identity and full-media provenance for one recording."""

    recording_session_id: str
    manifest_schema: str
    camera_serial: str
    frame_count: int
    manifest_sha256: str
    timestamps_sha256: str
    intrinsics_sha256: str
    rgb_media_sha256: str
    frame_provenance_sha256: str

    def __post_init__(self) -> None:
        for name in ("recording_session_id", "manifest_schema", "camera_serial"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty string without outer whitespace")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise TypeError("frame_count must be an integer")
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least two")
        for name in (
            "manifest_sha256",
            "timestamps_sha256",
            "intrinsics_sha256",
            "rgb_media_sha256",
            "frame_provenance_sha256",
        ):
            _validate_sha256(getattr(self, name), name=name)

    @property
    def content_fingerprint_sha256(self) -> str:
        """Identify captured content without trusting a mutable label or session name."""

        return _canonical_sha256(
            {
                "manifest_schema": self.manifest_schema,
                "camera_serial": self.camera_serial,
                "frame_count": self.frame_count,
                "timestamps_sha256": self.timestamps_sha256,
                "intrinsics_sha256": self.intrinsics_sha256,
                "rgb_media_sha256": self.rgb_media_sha256,
                "frame_provenance_sha256": self.frame_provenance_sha256,
            }
        )

    @property
    def recording_fingerprint_sha256(self) -> str:
        """Bind the capture session and immutable manifest to the captured content."""

        return _canonical_sha256(
            {
                "recording_session_id": self.recording_session_id,
                "manifest_sha256": self.manifest_sha256,
                "content_fingerprint_sha256": self.content_fingerprint_sha256,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "content_fingerprint_sha256": self.content_fingerprint_sha256,
            "recording_fingerprint_sha256": self.recording_fingerprint_sha256,
        }


@dataclass(frozen=True)
class PivotQualificationEvidence:
    """Independent labels and replay results for one robot-held pivot."""

    visible_frames: tuple[int, ...]
    tracked_visible_frames: tuple[int, ...]
    reviewed_axis_errors_deg: tuple[float, ...]
    reviewed_axis_label_frames: tuple[int, ...]
    automatic_reclamp_frames: tuple[int, ...]
    labelled_reclamp_frame: int
    earliest_allowed_reclamp_frame: int
    unresolved_reviewed_axis_label_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.visible_frames:
            raise ValueError("visible_frames cannot be empty")
        for name in (
            "visible_frames",
            "tracked_visible_frames",
            "reviewed_axis_label_frames",
            "automatic_reclamp_frames",
            "unresolved_reviewed_axis_label_frames",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing and unique")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"{name} must contain integer frame indices")
        visible = set(self.visible_frames)
        if not set(self.tracked_visible_frames).issubset(visible):
            raise ValueError("tracked_visible_frames must be a subset of visible_frames")
        if not set(self.reviewed_axis_label_frames).issubset(visible):
            raise ValueError("axis labels must lie in visible frames")
        if not set(self.unresolved_reviewed_axis_label_frames).issubset(visible):
            raise ValueError("unresolved axis labels must lie in visible frames")
        if set(self.reviewed_axis_label_frames) & set(self.unresolved_reviewed_axis_label_frames):
            raise ValueError("resolved and unresolved axis label frames cannot overlap")
        if len(self.reviewed_axis_errors_deg) != len(self.reviewed_axis_label_frames):
            raise ValueError("one finite error is required for every reviewed axis label")
        if not all(math.isfinite(value) for value in self.reviewed_axis_errors_deg):
            raise ValueError("reviewed axis errors must be finite")
        if self.earliest_allowed_reclamp_frame > self.labelled_reclamp_frame:
            raise ValueError("earliest allowed reclamp cannot follow the labelled reclamp")


@dataclass(frozen=True)
class NegativeCaseQualificationEvidence:
    """Tracker outputs over a human-labelled unobservable interval."""

    case_id: str
    kind: str
    evaluated_frames: tuple[int, ...]
    tracker_decisions: tuple[str, ...]
    gate_accepted_frames: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id cannot be empty")
        if self.kind not in REQUIRED_NEGATIVE_CASE_KINDS:
            raise ValueError(
                f"negative case kind must be one of {sorted(REQUIRED_NEGATIVE_CASE_KINDS)}"
            )
        if not self.evaluated_frames:
            raise ValueError("negative case must evaluate at least one frame")
        if tuple(sorted(set(self.evaluated_frames))) != self.evaluated_frames:
            raise ValueError("evaluated_frames must be strictly increasing and unique")
        if len(self.tracker_decisions) != len(self.evaluated_frames):
            raise ValueError("one tracker decision is required for every evaluated frame")
        if not set(self.gate_accepted_frames).issubset(self.evaluated_frames):
            raise ValueError("gate accepted frames must lie in the evaluated interval")


def _maximum_label_stride(frames: tuple[int, ...]) -> int | None:
    if len(frames) < 2:
        return None
    return max(second - first for first, second in zip(frames, frames[1:]))


def evaluate_robot_held_trial(
    *,
    trial_id: str,
    pivot: PivotQualificationEvidence,
    negative_cases: tuple[NegativeCaseQualificationEvidence, ...],
    criteria: RGBTrackerPromotionCriteria = CANONICAL_RGB_TRACKER_PROMOTION_CRITERIA,
) -> dict[str, object]:
    """Evaluate one trial without weakening or learning any threshold."""

    if not trial_id:
        raise ValueError("trial_id cannot be empty")

    visible_count = len(pivot.visible_frames)
    tracked_count = len(pivot.tracked_visible_frames)
    tracked_fraction = tracked_count / visible_count
    absolute_errors = tuple(abs(value) for value in pivot.reviewed_axis_errors_deg)
    error_count = len(absolute_errors)
    mae = sum(absolute_errors) / error_count if error_count else None
    maximum_error = max(absolute_errors) if absolute_errors else None
    first_reclamp = pivot.automatic_reclamp_frames[0] if pivot.automatic_reclamp_frames else None
    reclamp_frame_error = (
        None if first_reclamp is None else first_reclamp - pivot.labelled_reclamp_frame
    )
    false_early_frames = tuple(
        frame
        for frame in pivot.automatic_reclamp_frames
        if frame < pivot.earliest_allowed_reclamp_frame
    )
    all_label_frames = tuple(
        sorted((*pivot.reviewed_axis_label_frames, *pivot.unresolved_reviewed_axis_label_frames))
    )
    max_label_stride = _maximum_label_stride(all_label_frames)

    negative_reports: list[dict[str, object]] = []
    kinds_passed: set[str] = set()
    for case in negative_cases:
        invalid_decisions = [
            {"frame": frame, "decision": decision}
            for frame, decision in zip(case.evaluated_frames, case.tracker_decisions, strict=True)
            if decision not in FAIL_CLOSED_TRACKER_DECISIONS
        ]
        passed = not invalid_decisions and not case.gate_accepted_frames
        if passed:
            kinds_passed.add(case.kind)
        negative_reports.append(
            {
                "case_id": case.case_id,
                "kind": case.kind,
                "passed": passed,
                "evaluated_frame_count": len(case.evaluated_frames),
                "invalid_open_decisions": invalid_decisions,
                "gate_accepted_frames": list(case.gate_accepted_frames),
            }
        )

    checks = {
        "tracked_fraction": (
            tracked_fraction >= criteria.tracked_fraction_while_labelled_visible_min
        ),
        "reviewed_axis_label_count": (len(all_label_frames) >= criteria.min_reviewed_axis_labels),
        "all_reviewed_axis_labels_tracked": (not pivot.unresolved_reviewed_axis_label_frames),
        "reviewed_axis_label_stride": (
            max_label_stride is not None
            and max_label_stride <= criteria.max_axis_label_stride_frames
        ),
        "reviewed_axis_mae": (mae is not None and mae <= criteria.reviewed_axis_mae_deg_max),
        "reviewed_axis_max_error": (
            maximum_error is not None and maximum_error <= criteria.reviewed_axis_max_error_deg_max
        ),
        "reclamp_frame_error": (
            reclamp_frame_error is not None
            and abs(reclamp_frame_error) <= criteria.reclamp_frame_error_max
        ),
        "no_false_early_reclamp": (
            len(false_early_frames) <= criteria.false_reclamp_before_labelled_target_max
        ),
        "occlusion_fails_closed": (
            not criteria.require_occlusion_fail_closed or "partial_occlusion" in kinds_passed
        ),
        "ambiguity_fails_closed": (
            not criteria.require_ambiguity_fail_closed or "distractor_ambiguity" in kinds_passed
        ),
    }
    return {
        "trial_id": trial_id,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "visible_frame_count": visible_count,
            "tracked_visible_frame_count": tracked_count,
            "tracked_fraction_while_labelled_visible": tracked_fraction,
            "reviewed_axis_label_count": len(all_label_frames),
            "reviewed_axis_compared_count": error_count,
            "unresolved_reviewed_axis_label_frames": list(
                pivot.unresolved_reviewed_axis_label_frames
            ),
            "reviewed_axis_label_max_stride_frames": max_label_stride,
            "reviewed_axis_mae_deg": mae,
            "reviewed_axis_max_error_deg": maximum_error,
            "automatic_first_reclamp_frame": first_reclamp,
            "labelled_reclamp_frame": pivot.labelled_reclamp_frame,
            "earliest_allowed_reclamp_frame": pivot.earliest_allowed_reclamp_frame,
            "reclamp_frame_error": reclamp_frame_error,
            "false_early_reclamp_frames": list(false_early_frames),
        },
        "negative_cases": negative_reports,
    }


def evaluate_robot_held_bundle(
    trial_reports: tuple[dict[str, object], ...],
    *,
    recording_evidence: tuple[ImmutableRecordingEvidence, ...],
    criteria: RGBTrackerPromotionCriteria = CANONICAL_RGB_TRACKER_PROMOTION_CRITERIA,
) -> dict[str, object]:
    """Apply the three-distinct-trial promotion boundary."""

    if len(trial_reports) != len(recording_evidence):
        raise ValueError("one immutable recording identity is required for every trial")
    trial_ids = [str(report.get("trial_id", "")) for report in trial_reports]
    distinct_trial_ids = len(trial_ids) == len(set(trial_ids)) and all(trial_ids)
    session_ids = [evidence.recording_session_id for evidence in recording_evidence]
    content_fingerprints = [
        evidence.content_fingerprint_sha256 for evidence in recording_evidence
    ]
    recording_fingerprints = [
        evidence.recording_fingerprint_sha256 for evidence in recording_evidence
    ]
    distinct_session_ids = len(session_ids) == len(set(session_ids))
    distinct_content = len(content_fingerprints) == len(set(content_fingerprints))
    distinct_recordings = len(recording_fingerprints) == len(set(recording_fingerprints))
    successful = sum(report.get("passed") is True for report in trial_reports)
    enough_trials = len(trial_reports) >= criteria.successful_trials_required
    enough_successes = successful >= criteria.successful_trials_required
    all_provided_passed = successful == len(trial_reports)
    eligible = bool(
        distinct_trial_ids
        and distinct_session_ids
        and distinct_content
        and distinct_recordings
        and enough_trials
        and enough_successes
        and all_provided_passed
    )
    return {
        "runtime_gate_eligible": eligible,
        "successful_trials": successful,
        "successful_trials_required": criteria.successful_trials_required,
        "provided_trials": len(trial_reports),
        "all_trial_ids_distinct": distinct_trial_ids,
        "all_recording_session_ids_distinct": distinct_session_ids,
        "all_recording_content_fingerprints_distinct": distinct_content,
        "all_recording_fingerprints_distinct": distinct_recordings,
        "all_provided_trials_passed": all_provided_passed,
        "all_criteria_met": eligible,
    }


__all__ = [
    "CANONICAL_RGB_TRACKER_PROMOTION_CRITERIA",
    "FAIL_CLOSED_TRACKER_DECISIONS",
    "ImmutableRecordingEvidence",
    "NegativeCaseQualificationEvidence",
    "PivotQualificationEvidence",
    "REQUIRED_NEGATIVE_CASE_KINDS",
    "RGBTrackerPromotionCriteria",
    "evaluate_robot_held_bundle",
    "evaluate_robot_held_trial",
]
