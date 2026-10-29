#!/usr/bin/env python3
"""Qualify the test-tube front-RGB axis gate on recorded robot-held trials.

The input is a deterministic JSON annotation bundle plus lossless RGB frames.
This script never opens a camera, ROS, MoveIt, the gripper, or the robot.  It
does not update the runtime reference automatically; it only emits an evidence
report whose top-level ``runtime_gate_eligible`` field is true after three
distinct immutable recording sessions pass every fixed promotion criterion.
Distinctness is computed from the complete capture, independently of trial
identifiers, labels, ROIs, or replay intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from demo2gripper.test_tube_rgb_qualification import (  # noqa: E402
    CANONICAL_RGB_TRACKER_PROMOTION_CRITERIA,
    ImmutableRecordingEvidence,
    NegativeCaseQualificationEvidence,
    PivotQualificationEvidence,
    evaluate_robot_held_bundle,
    evaluate_robot_held_trial,
)
from demo2gripper.test_tube_rgb_tracker import (  # noqa: E402
    ImageLineCandidate,
    TubeAxisCandidateTracker,
    TubeAxisTrackResult,
    extract_hough_line_candidates,
)
from demo2gripper.test_tube_visual import (  # noqa: E402
    VisualAngleGate,
    VisualAngleGateConfig,
    axis_angle_error_deg,
)

ANNOTATION_SCHEMA = "demo2gripper.test_tube_rgb_qualification_annotations.v2"
REPORT_SCHEMA = "demo2gripper.test_tube_rgb_tracker_qualification.v2"
DEFAULT_REFERENCE = REPO / "configs/test_tube_front_rgb_pivot_reference_20261023.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_recording_path(value: object, *, annotation_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("run_dir must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = annotation_path.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"recorded trial directory not found: {path}")
    return path


def _line_from_seed(payload: object, *, expected_frame: int) -> ImageLineCandidate:
    if not isinstance(payload, dict):
        raise ValueError("seed must be an object")
    frame = payload.get("frame")
    if frame != expected_frame:
        raise ValueError("seed.frame must equal the replay start frame")
    line_uv = payload.get("line_uv")
    values = np.asarray(line_uv, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise ValueError("seed.line_uv must be a finite [[u0,v0],[u1,v1]] array")
    return ImageLineCandidate(tuple(values[0]), tuple(values[1]))


def _interval(payload: object, *, name: str, maximum_frame: int) -> tuple[int, int]:
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in payload)
    ):
        raise ValueError(f"{name} must contain two integer frame indices")
    start, end = payload
    if not 0 <= start <= end <= maximum_frame:
        raise ValueError(f"{name} lies outside the recording")
    return start, end


def _roi(payload: object, *, width: int, height: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(payload, list)
        or len(payload) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in payload)
    ):
        raise ValueError("roi_uvxy must contain four integers")
    u0, v0, u1, v1 = payload
    if not (0 <= u0 < u1 <= width and 0 <= v0 < v1 <= height):
        raise ValueError("roi_uvxy must lie inside the recorded image")
    return u0, v0, u1, v1


def _visible_frames(
    payload: object,
    *,
    replay_interval: tuple[int, int],
) -> tuple[int, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("visible_intervals_inclusive must be a non-empty array")
    start, end = replay_interval
    frames: list[int] = []
    for index, raw_interval in enumerate(payload):
        interval = _interval(
            raw_interval,
            name=f"visible_intervals_inclusive[{index}]",
            maximum_frame=end,
        )
        if interval[0] < start:
            raise ValueError("visible interval starts before pivot replay")
        frames.extend(range(interval[0], interval[1] + 1))
    if len(frames) != len(set(frames)):
        raise ValueError("visible intervals cannot overlap")
    return tuple(sorted(frames))


def _sorted_candidates(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
) -> list[ImageLineCandidate]:
    candidates = extract_hough_line_candidates(image, roi_uvxy=roi)
    return sorted(
        candidates,
        key=lambda line: (
            round(line.start_uv[0], 6),
            round(line.start_uv[1], 6),
            round(line.end_uv[0], 6),
            round(line.end_uv[1], 6),
        ),
    )


def _read_rgb(path: Path, *, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read RGB frame: {path}")
    if image.dtype != np.uint8 or image.shape != (height, width, 3):
        raise RuntimeError(f"unexpected RGB contract at {path}: {image.shape}/{image.dtype}")
    return image


def _make_gate(reference: dict[str, Any]) -> VisualAngleGate:
    target = reference["target"]
    return VisualAngleGate(
        VisualAngleGateConfig(
            target_image_axis_deg=float(target["target_image_axis_deg"]),
            tolerance_deg=float(target["tolerance_deg"]),
            min_consecutive_frames=int(target["min_consecutive_frames"]),
            max_rate_deg_s=float(target["max_rate_deg_s"]),
        )
    )


def _seed_result(line: ImageLineCandidate, observation: Any) -> TubeAxisTrackResult:
    return TubeAxisTrackResult(
        decision="REVIEWED_SEED",
        observation=observation,
        selected_line=line,
        considered_candidate_count=1,
        equivalent_candidate_count=1,
        tracking_score=1.0,
        ambiguity_margin=1.0,
        reason="independently reviewed robot-held tube edge",
    )


def _replay_pivot(
    *,
    run_dir: Path,
    timestamps: np.ndarray,
    width: int,
    height: int,
    pivot: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[PivotQualificationEvidence, list[dict[str, object]], set[Path]]:
    maximum_frame = len(timestamps) - 1
    start, end = _interval(
        pivot.get("frame_interval_inclusive"),
        name="pivot.frame_interval_inclusive",
        maximum_frame=maximum_frame,
    )
    roi = _roi(pivot.get("roi_uvxy"), width=width, height=height)
    seed_line = _line_from_seed(pivot.get("seed"), expected_frame=start)
    visible_frames = _visible_frames(
        pivot.get("visible_intervals_inclusive"), replay_interval=(start, end)
    )
    visible_set = set(visible_frames)

    raw_labels = pivot.get("reviewed_axis_labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("pivot.reviewed_axis_labels must be a non-empty array")
    reviewed: dict[int, float] = {}
    for label in raw_labels:
        if not isinstance(label, dict):
            raise ValueError("each reviewed axis label must be an object")
        frame = label.get("frame")
        angle = label.get("axis_angle_deg")
        if (
            isinstance(frame, bool)
            or not isinstance(frame, int)
            or frame not in visible_set
            or isinstance(angle, bool)
            or not isinstance(angle, (int, float))
            or not math.isfinite(float(angle))
        ):
            raise ValueError("reviewed label needs a visible frame and finite axis angle")
        if frame in reviewed:
            raise ValueError("reviewed axis label frames must be unique")
        reviewed[frame] = float(angle) % 180.0

    labelled_reclamp = pivot.get("labelled_reclamp_frame")
    earliest_allowed = pivot.get("earliest_allowed_reclamp_frame", labelled_reclamp)
    for name, value in (
        ("labelled_reclamp_frame", labelled_reclamp),
        ("earliest_allowed_reclamp_frame", earliest_allowed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not start <= value <= end:
            raise ValueError(f"pivot.{name} must lie inside the replay interval")

    tracker = TubeAxisCandidateTracker()
    gate = _make_gate(reference)
    seed_observation = tracker.seed(timestamp_s=float(timestamps[start]), line=seed_line)
    gate_result = gate.update(timestamp_s=float(timestamps[start]), observation=seed_observation)
    seed_result = _seed_result(seed_line, seed_observation)

    tracked_visible: list[int] = []
    resolved_label_frames: list[int] = []
    unresolved_label_frames: list[int] = []
    errors: list[float] = []
    accepted_frames: list[int] = []
    rows: list[dict[str, object]] = []
    rgb_paths: set[Path] = set()
    for frame in range(start, end + 1):
        rgb_path = run_dir / "rgb" / f"{frame:06d}.png"
        image = _read_rgb(rgb_path, width=width, height=height)
        rgb_paths.add(rgb_path)
        if frame == start:
            result = seed_result
            candidates = [seed_line]
        else:
            candidates = _sorted_candidates(image, roi)
            result = tracker.update(
                timestamp_s=float(timestamps[frame]),
                candidates=candidates,
                roi_uvxy=roi,
            )
            gate_result = gate.update(
                timestamp_s=float(timestamps[frame]), observation=result.observation
            )
        if frame in visible_set and result.observation.valid:
            tracked_visible.append(frame)
        if gate_result.accepted:
            accepted_frames.append(frame)
        automatic_angle = result.observation.axis_angle_deg
        reviewed_angle = reviewed.get(frame)
        error: float | None = None
        if reviewed_angle is not None:
            if automatic_angle is None:
                unresolved_label_frames.append(frame)
            else:
                resolved_label_frames.append(frame)
                error = axis_angle_error_deg(automatic_angle, reviewed_angle)
                errors.append(error)
        rows.append(
            {
                "frame": frame,
                "timestamp_s": float(timestamps[frame]),
                "human_label_state": "visible" if frame in visible_set else "excluded",
                "candidate_count": len(candidates),
                "tracker_decision": result.decision,
                "axis_valid": int(result.observation.valid),
                "automatic_axis_deg": "" if automatic_angle is None else automatic_angle,
                "reviewed_axis_deg": "" if reviewed_angle is None else reviewed_angle,
                "axis_error_deg": "" if error is None else error,
                "gate_decision": gate_result.decision,
                "gate_accepted": int(gate_result.accepted),
            }
        )

    evidence = PivotQualificationEvidence(
        visible_frames=visible_frames,
        tracked_visible_frames=tuple(tracked_visible),
        reviewed_axis_errors_deg=tuple(errors),
        reviewed_axis_label_frames=tuple(resolved_label_frames),
        automatic_reclamp_frames=tuple(accepted_frames),
        labelled_reclamp_frame=labelled_reclamp,
        earliest_allowed_reclamp_frame=earliest_allowed,
        unresolved_reviewed_axis_label_frames=tuple(unresolved_label_frames),
    )
    return evidence, rows, rgb_paths


def _replay_negative_case(
    *,
    run_dir: Path,
    timestamps: np.ndarray,
    width: int,
    height: int,
    payload: dict[str, Any],
    reference: dict[str, Any],
) -> tuple[
    NegativeCaseQualificationEvidence,
    list[dict[str, object]],
    set[Path],
]:
    maximum_frame = len(timestamps) - 1
    start, end = _interval(
        payload.get("replay_interval_inclusive"),
        name="negative_case.replay_interval_inclusive",
        maximum_frame=maximum_frame,
    )
    roi = _roi(payload.get("roi_uvxy"), width=width, height=height)
    seed_line = _line_from_seed(payload.get("seed"), expected_frame=start)
    raw_evaluation_frames = payload.get("evaluation_frames")
    if not isinstance(raw_evaluation_frames, list) or not raw_evaluation_frames:
        raise ValueError("negative case evaluation_frames must be a non-empty array")
    evaluation_frames = tuple(raw_evaluation_frames)
    if tuple(sorted(set(evaluation_frames))) != evaluation_frames or any(
        isinstance(frame, bool) or not isinstance(frame, int) or not start < frame <= end
        for frame in evaluation_frames
    ):
        raise ValueError("negative evaluation frames must be unique and inside replay")

    case_id = payload.get("case_id")
    kind = payload.get("kind")
    if not isinstance(case_id, str) or not isinstance(kind, str):
        raise ValueError("negative case needs string case_id and kind")

    tracker = TubeAxisCandidateTracker()
    gate = _make_gate(reference)
    seed_observation = tracker.seed(timestamp_s=float(timestamps[start]), line=seed_line)
    gate.update(timestamp_s=float(timestamps[start]), observation=seed_observation)
    evaluated_decisions: list[str] = []
    gate_accepted_frames: list[int] = []
    rows: list[dict[str, object]] = []
    rgb_paths: set[Path] = set()
    evaluation_set = set(evaluation_frames)
    for frame in range(start + 1, end + 1):
        rgb_path = run_dir / "rgb" / f"{frame:06d}.png"
        image = _read_rgb(rgb_path, width=width, height=height)
        rgb_paths.add(rgb_path)
        candidates = _sorted_candidates(image, roi)
        result = tracker.update(
            timestamp_s=float(timestamps[frame]),
            candidates=candidates,
            roi_uvxy=roi,
        )
        gate_result = gate.update(
            timestamp_s=float(timestamps[frame]), observation=result.observation
        )
        evaluated = frame in evaluation_set
        if evaluated:
            evaluated_decisions.append(result.decision)
            if gate_result.accepted:
                gate_accepted_frames.append(frame)
        rows.append(
            {
                "frame": frame,
                "timestamp_s": float(timestamps[frame]),
                "human_label_state": kind if evaluated else "replay_context",
                "candidate_count": len(candidates),
                "tracker_decision": result.decision,
                "axis_valid": int(result.observation.valid),
                "automatic_axis_deg": ""
                if result.observation.axis_angle_deg is None
                else result.observation.axis_angle_deg,
                "gate_decision": gate_result.decision,
                "gate_accepted": int(gate_result.accepted),
            }
        )

    evidence = NegativeCaseQualificationEvidence(
        case_id=case_id,
        kind=kind,
        evaluated_frames=evaluation_frames,
        tracker_decisions=tuple(evaluated_decisions),
        gate_accepted_frames=tuple(gate_accepted_frames),
    )
    return evidence, rows, rgb_paths


def _verify_recording(
    *,
    run_dir: Path,
    reference: dict[str, Any],
    expected_recording_session_id: str,
) -> tuple[np.ndarray, int, int, ImmutableRecordingEvidence]:
    manifest_path = run_dir / "input_manifest.json"
    timestamp_path = run_dir / "timestamps_s.npy"
    intrinsics_path = run_dir / "K.npy"
    manifest = _read_json(manifest_path)
    recording_session_id = manifest.get("recording_session_id")
    if not isinstance(recording_session_id, str) or not recording_session_id:
        raise ValueError("recording manifest lacks a non-empty recording_session_id")
    if recording_session_id != expected_recording_session_id:
        raise ValueError("trial recording_session_id does not match the recording manifest")
    timestamps = np.load(timestamp_path, allow_pickle=False)
    if (
        timestamps.ndim != 1
        or len(timestamps) < 2
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("timestamps_s.npy must be a finite, strictly increasing vector")
    camera = reference["camera"]
    if manifest.get("device", {}).get("serial") != camera["serial"]:
        raise ValueError("recorded camera serial does not match the reference")
    intrinsics_sha = _sha256(intrinsics_path)
    if intrinsics_sha != camera["intrinsics_sha256"]:
        raise ValueError("recorded K.npy does not match the reference intrinsics")
    resolution = camera["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("reference camera resolution is malformed")
    width, height = int(resolution[0]), int(resolution[1])
    stream = manifest.get("stream", {})
    if stream.get("width") != width or stream.get("height") != height:
        raise ValueError("recorded resolution does not match the reference")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != len(timestamps):
        raise ValueError("manifest frames must match the complete timestamp vector")
    rgb_media_entries: list[dict[str, str]] = []
    frame_provenance: list[dict[str, object]] = []
    listed_rgb_paths: set[Path] = set()
    previous_frame_number: int | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("index") != index:
            raise ValueError("manifest frame indices must be contiguous from zero")
        relative_rgb = frame.get("rgb")
        if not isinstance(relative_rgb, str) or not relative_rgb:
            raise ValueError("every manifest frame must name its RGB file")
        relative_path = Path(relative_rgb)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("manifest RGB paths must stay inside the recording directory")
        rgb_path = (run_dir / relative_path).resolve()
        if not rgb_path.is_relative_to(run_dir) or not rgb_path.is_file():
            raise ValueError(
                f"manifest RGB file is missing or escapes the recording: {relative_rgb}"
            )
        if rgb_path in listed_rgb_paths:
            raise ValueError("manifest RGB paths must be unique")
        listed_rgb_paths.add(rgb_path)

        color_sensor_timestamp_s = frame.get("color_sensor_timestamp_s")
        color_sdk_timestamp_s = frame.get("color_sdk_timestamp_s")
        sdk_timestamp_domain = frame.get("sdk_timestamp_domain")
        color_frame_number = frame.get("color_frame_number")
        if (
            isinstance(color_sensor_timestamp_s, bool)
            or not isinstance(color_sensor_timestamp_s, (int, float))
            or not math.isfinite(float(color_sensor_timestamp_s))
        ):
            raise ValueError("every frame needs a finite color sensor timestamp")
        if not math.isclose(
            float(color_sensor_timestamp_s),
            float(timestamps[index]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("manifest color timestamps do not match timestamps_s.npy")
        if isinstance(color_frame_number, bool) or not isinstance(color_frame_number, int):
            raise ValueError("every frame needs an integer color frame number")
        if (
            isinstance(color_sdk_timestamp_s, bool)
            or not isinstance(color_sdk_timestamp_s, (int, float))
            or not math.isfinite(float(color_sdk_timestamp_s))
        ):
            raise ValueError("every frame needs a finite color SDK timestamp")
        if not isinstance(sdk_timestamp_domain, str) or not sdk_timestamp_domain:
            raise ValueError("every frame needs a non-empty SDK timestamp domain")
        if previous_frame_number is not None and color_frame_number <= previous_frame_number:
            raise ValueError("manifest color frame numbers must be strictly increasing")
        previous_frame_number = color_frame_number

        rgb_media_entries.append(
            {"relative_path": relative_path.as_posix(), "sha256": _sha256(rgb_path)}
        )
        frame_provenance.append(
            {
                "index": index,
                "rgb": relative_path.as_posix(),
                "color_sensor_timestamp_s": float(color_sensor_timestamp_s),
                "color_sdk_timestamp_s": float(color_sdk_timestamp_s),
                "sdk_timestamp_domain": sdk_timestamp_domain,
                "color_frame_number": color_frame_number,
            }
        )

    actual_rgb_paths = {
        path.resolve() for path in (run_dir / "rgb").iterdir() if path.is_file()
    }
    if actual_rgb_paths != listed_rgb_paths:
        raise ValueError("manifest must enumerate every file in the recording RGB directory")

    manifest_schema = manifest.get("schema")
    camera_serial = manifest.get("device", {}).get("serial")
    if not isinstance(manifest_schema, str) or not manifest_schema:
        raise ValueError("recording manifest lacks a schema")
    if not isinstance(camera_serial, str) or not camera_serial:
        raise ValueError("recording manifest lacks a camera serial")
    identity = ImmutableRecordingEvidence(
        recording_session_id=recording_session_id,
        manifest_schema=manifest_schema,
        camera_serial=camera_serial,
        frame_count=len(frames),
        manifest_sha256=_sha256(manifest_path),
        timestamps_sha256=_sha256(timestamp_path),
        intrinsics_sha256=intrinsics_sha,
        rgb_media_sha256=_canonical_sha256(rgb_media_entries),
        frame_provenance_sha256=_canonical_sha256(frame_provenance),
    )
    return timestamps, width, height, identity


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty replay table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rgb_digest(paths: set[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def qualify(
    *,
    annotation_path: Path,
    reference_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    annotation_path = annotation_path.resolve()
    reference_path = reference_path.resolve()
    output_path = output_path.resolve()
    annotations = _read_json(annotation_path)
    reference = _read_json(reference_path)
    if annotations.get("schema") != ANNOTATION_SCHEMA:
        raise ValueError(f"expected annotation schema {ANNOTATION_SCHEMA}")
    qualification_id = annotations.get("qualification_id")
    if not isinstance(qualification_id, str) or not qualification_id:
        raise ValueError("annotations.qualification_id must be a non-empty string")
    trials = annotations.get("trials")
    if not isinstance(trials, list):
        raise ValueError("annotations.trials must be an array")

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_reports: list[dict[str, object]] = []
    recording_evidence: list[ImmutableRecordingEvidence] = []
    artifact_tables: dict[str, dict[str, object]] = {}
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("each trial must be an object")
        trial_id = trial.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("each trial needs a non-empty trial_id")
        allowed_trial_id_characters = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        )
        if any(character not in allowed_trial_id_characters for character in trial_id):
            raise ValueError("trial_id may contain only letters, digits, '_' and '-'")
        recording_session_id = trial.get("recording_session_id")
        if (
            not isinstance(recording_session_id, str)
            or not recording_session_id
            or recording_session_id != recording_session_id.strip()
            or len(recording_session_id) > 128
        ):
            raise ValueError(
                "trial.recording_session_id must be a non-empty string without outer whitespace"
            )
        run_dir = _resolve_recording_path(trial.get("run_dir"), annotation_path=annotation_path)
        timestamps, width, height, identity = _verify_recording(
            run_dir=run_dir,
            reference=reference,
            expected_recording_session_id=recording_session_id,
        )
        pivot_payload = trial.get("pivot")
        if not isinstance(pivot_payload, dict):
            raise ValueError("trial.pivot must be an object")
        pivot_evidence, pivot_rows, rgb_paths = _replay_pivot(
            run_dir=run_dir,
            timestamps=timestamps,
            width=width,
            height=height,
            pivot=pivot_payload,
            reference=reference,
        )

        raw_negative_cases = trial.get("negative_cases")
        if not isinstance(raw_negative_cases, list):
            raise ValueError("trial.negative_cases must be an array")
        negative_evidence: list[NegativeCaseQualificationEvidence] = []
        negative_tables: dict[str, str] = {}
        for negative_payload in raw_negative_cases:
            if not isinstance(negative_payload, dict):
                raise ValueError("each negative case must be an object")
            evidence, rows, case_rgb_paths = _replay_negative_case(
                run_dir=run_dir,
                timestamps=timestamps,
                width=width,
                height=height,
                payload=negative_payload,
                reference=reference,
            )
            negative_evidence.append(evidence)
            rgb_paths.update(case_rgb_paths)
            table_path = output_dir / f"{trial_id}_{evidence.case_id}.csv"
            _write_csv(table_path, rows)
            negative_tables[evidence.case_id] = str(table_path)

        report = evaluate_robot_held_trial(
            trial_id=trial_id,
            pivot=pivot_evidence,
            negative_cases=tuple(negative_evidence),
        )
        pivot_table_path = output_dir / f"{trial_id}_pivot.csv"
        _write_csv(pivot_table_path, pivot_rows)
        rgb_sha = _rgb_digest(rgb_paths)
        report["recording"] = {
            "run_dir": str(run_dir),
            **identity.to_dict(),
            "selected_rgb_evidence_sha256": rgb_sha,
            "distinctness_basis": (
                "full recording identity and media provenance; annotations, trial_id, "
                "and selected replay intervals are excluded"
            ),
        }
        report["artifacts"] = {
            "pivot_frame_table": str(pivot_table_path),
            "negative_case_frame_tables": negative_tables,
        }
        trial_reports.append(report)
        recording_evidence.append(identity)
        artifact_tables[trial_id] = report["artifacts"]

    aggregate = evaluate_robot_held_bundle(
        tuple(trial_reports), recording_evidence=tuple(recording_evidence)
    )
    eligible = aggregate["runtime_gate_eligible"] is True
    report_payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": (
            "qualified_for_runtime_gate" if eligible else "qualification_failed_or_incomplete"
        ),
        "runtime_gate_eligible": eligible,
        "hardware_commands_dispatched": False,
        "camera_opened": False,
        "recorded_rgb_only": True,
        "depth_used_for_tube_axis": False,
        "qualification_id": qualification_id,
        "source": {
            "annotations": str(annotation_path),
            "annotations_sha256": _sha256(annotation_path),
            "reference_config": str(reference_path),
            "reference_config_sha256": _sha256(reference_path),
            "camera_serial": reference["camera"]["serial"],
            "intrinsics_sha256": reference["camera"]["intrinsics_sha256"],
        },
        "promotion_criteria": CANONICAL_RGB_TRACKER_PROMOTION_CRITERIA.to_dict(),
        "aggregate": aggregate,
        "trials": trial_reports,
        "integration_boundary": {
            "promote_only_when": "runtime_gate_eligible is true in this immutable report",
            "reference_update_is_automatic": False,
            "preflight_input": (
                "Pin this report path and SHA-256 as the robot-held front-RGB tracker "
                "qualification. Do not promote the human-held demo replay."
            ),
            "runtime_still_requires": (
                "the same camera serial/intrinsics/mount, a fresh reviewed seed, and "
                "fresh fail-closed tracker evidence"
            ),
        },
        "artifacts": artifact_tables,
    }
    output_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    return report_payload


def main() -> int:
    args = parse_args()
    report = qualify(
        annotation_path=args.annotations,
        reference_path=args.reference_config,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["runtime_gate_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
