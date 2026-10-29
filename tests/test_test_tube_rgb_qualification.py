from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from demo2gripper.test_tube_rgb_qualification import (
    ImmutableRecordingEvidence,
    NegativeCaseQualificationEvidence,
    PivotQualificationEvidence,
    evaluate_robot_held_bundle,
    evaluate_robot_held_trial,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import qualify_test_tube_rgb_tracker as qualifier  # noqa: E402


def _recording(
    suffix: str,
    *,
    session_id: str | None = None,
    rgb_digest: str | None = None,
) -> ImmutableRecordingEvidence:
    digit = suffix[-1]
    return ImmutableRecordingEvidence(
        recording_session_id=session_id or f"recording-session-{suffix}",
        manifest_schema="demo2gripper.realsense_rgbd_raw_tactile.v1",
        camera_serial="254622073804",
        frame_count=50,
        manifest_sha256=digit * 64,
        timestamps_sha256=chr(ord(digit) + 1) * 64,
        intrinsics_sha256="a" * 64,
        rgb_media_sha256=rgb_digest or chr(ord(digit) + 2) * 64,
        frame_provenance_sha256=chr(ord(digit) + 3) * 64,
    )


def _pivot() -> PivotQualificationEvidence:
    frames = tuple(range(21))
    return PivotQualificationEvidence(
        visible_frames=frames,
        tracked_visible_frames=frames,
        reviewed_axis_errors_deg=(0.5, -1.0, 1.5, -2.0, 0.0),
        reviewed_axis_label_frames=(0, 5, 10, 15, 20),
        automatic_reclamp_frames=(20,),
        labelled_reclamp_frame=20,
        earliest_allowed_reclamp_frame=18,
    )


def _negative_cases() -> tuple[NegativeCaseQualificationEvidence, ...]:
    return (
        NegativeCaseQualificationEvidence(
            case_id="occ",
            kind="partial_occlusion",
            evaluated_frames=(22, 23),
            tracker_decisions=("HOLD_AND_REOBSERVE", "RESEED_REQUIRED"),
        ),
        NegativeCaseQualificationEvidence(
            case_id="amb",
            kind="distractor_ambiguity",
            evaluated_frames=(25, 26),
            tracker_decisions=("HOLD_AND_REOBSERVE", "HOLD_AND_REOBSERVE"),
        ),
    )


def _pass_report(trial_id: str) -> dict[str, object]:
    return evaluate_robot_held_trial(
        trial_id=trial_id,
        pivot=_pivot(),
        negative_cases=_negative_cases(),
    )


def test_robot_held_trial_passes_all_fixed_criteria() -> None:
    report = _pass_report("trial_01")
    assert report["passed"] is True
    assert all(report["checks"].values())
    metrics = report["metrics"]
    assert metrics["tracked_fraction_while_labelled_visible"] == pytest.approx(1.0)
    assert metrics["reviewed_axis_mae_deg"] == pytest.approx(1.0)
    assert metrics["reviewed_axis_max_error_deg"] == pytest.approx(2.0)
    assert metrics["reclamp_frame_error"] == 0


def test_missing_label_observation_fails_instead_of_being_dropped() -> None:
    pivot = replace(
        _pivot(),
        reviewed_axis_errors_deg=(0.5, -1.0, -2.0, 0.0),
        reviewed_axis_label_frames=(0, 5, 15, 20),
        unresolved_reviewed_axis_label_frames=(10,),
    )
    report = evaluate_robot_held_trial(
        trial_id="trial_01",
        pivot=pivot,
        negative_cases=_negative_cases(),
    )
    assert report["passed"] is False
    assert report["checks"]["all_reviewed_axis_labels_tracked"] is False
    assert report["metrics"]["unresolved_reviewed_axis_label_frames"] == [10]


@pytest.mark.parametrize(
    ("pivot", "failed_check"),
    [
        (
            replace(
                _pivot(),
                tracked_visible_frames=tuple(range(19)),
            ),
            "tracked_fraction",
        ),
        (
            replace(
                _pivot(),
                reviewed_axis_errors_deg=(0.0, 0.0, 0.0, 0.0, 5.1),
            ),
            "reviewed_axis_max_error",
        ),
        (
            replace(_pivot(), automatic_reclamp_frames=(16, 20)),
            "no_false_early_reclamp",
        ),
        (
            replace(_pivot(), automatic_reclamp_frames=(16,)),
            "reclamp_frame_error",
        ),
    ],
)
def test_promotion_metrics_fail_closed(
    pivot: PivotQualificationEvidence,
    failed_check: str,
) -> None:
    report = evaluate_robot_held_trial(
        trial_id="trial_01",
        pivot=pivot,
        negative_cases=_negative_cases(),
    )
    assert report["passed"] is False
    assert report["checks"][failed_check] is False


def test_non_fail_closed_negative_case_blocks_trial() -> None:
    cases = (
        _negative_cases()[0],
        replace(
            _negative_cases()[1],
            tracker_decisions=("TRACKED", "HOLD_AND_REOBSERVE"),
        ),
    )
    report = evaluate_robot_held_trial(
        trial_id="trial_01",
        pivot=_pivot(),
        negative_cases=cases,
    )
    assert report["passed"] is False
    assert report["checks"]["ambiguity_fails_closed"] is False
    ambiguity = report["negative_cases"][1]
    assert ambiguity["invalid_open_decisions"] == [{"frame": 25, "decision": "TRACKED"}]


def test_bundle_requires_three_distinct_passing_trials_and_evidence() -> None:
    reports = tuple(_pass_report(f"trial_{index}") for index in range(3))
    passed = evaluate_robot_held_bundle(
        reports,
        recording_evidence=(_recording("1"), _recording("2"), _recording("3")),
    )
    assert passed["runtime_gate_eligible"] is True

    same_recording = _recording("1")
    relabelled = evaluate_robot_held_bundle(
        reports,
        recording_evidence=(same_recording, same_recording, same_recording),
    )
    assert relabelled["runtime_gate_eligible"] is False
    assert relabelled["all_recording_session_ids_distinct"] is False
    assert relabelled["all_recording_content_fingerprints_distinct"] is False
    assert relabelled["all_recording_fingerprints_distinct"] is False

    repeated = evaluate_robot_held_bundle(
        reports,
        recording_evidence=(
            _recording("1", session_id="first"),
            _recording("1", session_id="second"),
            _recording("3", session_id="third"),
        ),
    )
    assert repeated["runtime_gate_eligible"] is False
    assert repeated["all_recording_session_ids_distinct"] is True
    assert repeated["all_recording_content_fingerprints_distinct"] is False


def test_bundle_rejects_an_extra_failed_trial() -> None:
    reports = tuple(_pass_report(f"trial_{index}") for index in range(3))
    failed = dict(_pass_report("trial_3"))
    failed["passed"] = False
    bundle = evaluate_robot_held_bundle(
        (*reports, failed),
        recording_evidence=tuple(_recording(str(index)) for index in range(1, 5)),
    )
    assert bundle["successful_trials"] == 3
    assert bundle["all_provided_trials_passed"] is False
    assert bundle["runtime_gate_eligible"] is False


def _write_duplicate_recording(
    path: Path,
    *,
    recording_session_id: str,
    k: np.ndarray,
) -> None:
    (path / "rgb").mkdir(parents=True)
    np.save(path / "K.npy", k, allow_pickle=False)
    timestamps = np.asarray([10.0, 10.1], dtype=np.float64)
    np.save(path / "timestamps_s.npy", timestamps, allow_pickle=False)
    for index in range(2):
        (path / "rgb" / f"{index:06d}.png").write_bytes(
            b"identical-recorded-rgb-contents-" + bytes([index])
        )
    manifest = {
        "schema": "demo2gripper.realsense_rgbd_raw_tactile.v1",
        "recording_session_id": recording_session_id,
        "device": {"serial": "camera-serial"},
        "stream": {"width": 640, "height": 480},
        "frames": [
            {
                "index": index,
                "rgb": f"rgb/{index:06d}.png",
                "color_sensor_timestamp_s": float(timestamps[index]),
                "color_sdk_timestamp_s": 1_000.0 + float(timestamps[index]),
                "sdk_timestamp_domain": "global_time",
                "color_frame_number": 100 + index,
            }
            for index in range(2)
        ],
    }
    (path / "input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_cli_rejects_same_media_relabelled_as_three_recordings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    k = np.eye(3, dtype=np.float64)
    run_dirs: list[Path] = []
    session_ids = ["edited-session-a", "edited-session-b", "edited-session-c"]
    for index, session_id in enumerate(session_ids):
        run_dir = tmp_path / f"copy_{index}"
        _write_duplicate_recording(run_dir, recording_session_id=session_id, k=k)
        run_dirs.append(run_dir)

    reference_path = tmp_path / "reference.json"
    reference = {
        "runtime_gate_eligible": False,
        "camera": {
            "serial": "camera-serial",
            "intrinsics_sha256": qualifier._sha256(run_dirs[0] / "K.npy"),
            "resolution": [640, 480],
        },
    }
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    annotations = {
        "schema": qualifier.ANNOTATION_SCHEMA,
        "qualification_id": "duplicate-media-adversarial",
        "trials": [
            {
                "trial_id": f"different_trial_{index}",
                "recording_session_id": session_id,
                "run_dir": str(run_dir),
                "pivot": {"deliberately_different_annotation": index},
                "negative_cases": [
                    {"case_id": f"occ_{index}", "kind": "partial_occlusion"},
                    {"case_id": f"amb_{index}", "kind": "distractor_ambiguity"},
                ],
            }
            for index, (session_id, run_dir) in enumerate(zip(session_ids, run_dirs, strict=True))
        ],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

    def replay_pivot(**kwargs: object) -> tuple[object, list[dict[str, object]], set[Path]]:
        run_dir = Path(str(kwargs["run_dir"]))
        return _pivot(), [{"frame": 0}], {run_dir / "rgb/000000.png"}

    def replay_negative(
        **kwargs: object,
    ) -> tuple[object, list[dict[str, object]], set[Path]]:
        run_dir = Path(str(kwargs["run_dir"]))
        payload = kwargs["payload"]
        assert isinstance(payload, dict)
        evidence = NegativeCaseQualificationEvidence(
            case_id=str(payload["case_id"]),
            kind=str(payload["kind"]),
            evaluated_frames=(1,),
            tracker_decisions=("HOLD_AND_REOBSERVE",),
        )
        return evidence, [{"frame": 1}], {run_dir / "rgb/000001.png"}

    monkeypatch.setattr(qualifier, "_replay_pivot", replay_pivot)
    monkeypatch.setattr(qualifier, "_replay_negative_case", replay_negative)
    output_path = tmp_path / "qualification.json"
    report = qualifier.qualify(
        annotation_path=annotation_path,
        reference_path=reference_path,
        output_path=output_path,
    )

    assert all(trial["passed"] for trial in report["trials"])
    assert report["aggregate"]["all_trial_ids_distinct"] is True
    assert report["aggregate"]["all_recording_session_ids_distinct"] is True
    assert report["aggregate"]["all_recording_fingerprints_distinct"] is True
    assert report["aggregate"]["all_recording_content_fingerprints_distinct"] is False
    assert report["runtime_gate_eligible"] is False
    assert len(
        {
            trial["recording"]["content_fingerprint_sha256"]
            for trial in report["trials"]
        }
    ) == 1
    assert json.loads(reference_path.read_text())["runtime_gate_eligible"] is False
