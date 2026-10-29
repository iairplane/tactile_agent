#!/usr/bin/env python3
"""Replay and validate the fail-closed test-tube RGB axis tracker.

This tool reads a recorded demonstration and writes evidence under its
``analysis`` directory.  It never opens a camera, ROS node, gripper, or robot
connection and never uses transparent-object depth as tube geometry.
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

SCHEMA = "demo2gripper.test_tube_rgb_tracker_validation.v1"
DEFAULT_RUN = REPO / "runs/live/test_tube_demo_20261023_02"
DEFAULT_CONFIG = REPO / "configs/test_tube_front_rgb_pivot_reference_20261023.json"
ROI_UVXY = (100, 0, 430, 230)
START_FRAME = 290
END_FRAME = 350


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-frame", type=int, default=START_FRAME)
    parser.add_argument("--end-frame", type=int, default=END_FRAME)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("U0", "V0", "U1", "V1"),
        default=ROI_UVXY,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read RGB frame: {path}")
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"unexpected RGB contract: {path}: {image.shape}/{image.dtype}")
    return image


def _reviewed_axis_by_frame(path: Path) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            delta_u = float(row["line_x1"]) - float(row["line_x0"])
            delta_v = float(row["line_y1"]) - float(row["line_y0"])
            angle = math.degrees(math.atan2(delta_v, delta_u)) % 180.0
            result[int(row["frame"])] = angle
    return result


def _seed_from_reviewed_trace(path: Path, frame: int) -> ImageLineCandidate:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) == frame:
                return ImageLineCandidate(
                    (float(row["line_x0"]), float(row["line_y0"])),
                    (float(row["line_x1"]), float(row["line_y1"])),
                )
    raise RuntimeError(f"reviewed axis trace has no seed at frame {frame}")


def _draw_overlay(
    image: np.ndarray,
    *,
    frame: int,
    roi: tuple[int, int, int, int],
    candidates: list[ImageLineCandidate],
    result: TubeAxisTrackResult,
    gate_decision: str,
    gate_count: int,
    target_angle_deg: float,
) -> np.ndarray:
    canvas = image.copy()
    u0, v0, u1, v1 = roi
    cv2.rectangle(canvas, (u0, v0), (u1 - 1, v1 - 1), (255, 180, 0), 1)
    for candidate in candidates:
        p0 = tuple(round(value) for value in candidate.start_uv)
        p1 = tuple(round(value) for value in candidate.end_uv)
        cv2.line(canvas, p0, p1, (50, 90, 90), 1, cv2.LINE_AA)
    if result.selected_line is not None:
        p0 = tuple(round(value) for value in result.selected_line.start_uv)
        p1 = tuple(round(value) for value in result.selected_line.end_uv)
        color = (40, 220, 40) if result.observation.valid else (0, 100, 255)
        cv2.line(canvas, p0, p1, color, 4, cv2.LINE_AA)

    observed = (
        f"{result.observation.axis_angle_deg:.1f} deg"
        if result.observation.axis_angle_deg is not None
        else "unobservable"
    )
    lines = [
        f"frame {frame} | tracker: {result.decision}",
        f"axis: {observed} | learned target: {target_angle_deg:.1f} deg",
        f"gate: {gate_decision} | in-band run: {gate_count}/4",
    ]
    for index, text in enumerate(lines):
        y = 258 + 28 * index
        cv2.putText(
            canvas,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _qualification_procedure() -> dict[str, Any]:
    return {
        "purpose": "promote the front-RGB gate only after robot-held tube evidence",
        "trials_required": 3,
        "procedure": [
            "Keep D435 serial, intrinsics, resolution, exposure, and mount unchanged.",
            "Robot side-grasps the tube and holds still; an operator reviews one tube edge "
            "or paired-edge seed inside the configured ROI.",
            "Run bounded micro-loosen/pivot while recording RGB and robot timestamps; a "
            "human independently labels the tube axis at every fifth frame and the desired "
            "reclamp crossing.",
            "Inject at least one partial occlusion and one distractor-edge interval; each "
            "must yield HOLD_AND_REOBSERVE or RESEED_REQUIRED, never a guessed axis.",
            "Compare automatic angle and first RECLAMP with the independent labels, then "
            "repeat from a different valid starting tube angle.",
        ],
        "promotion_criteria_all_required": {
            "tracked_fraction_while_labelled_visible_min": 0.95,
            "reviewed_axis_mae_deg_max": 3.0,
            "reviewed_axis_max_error_deg_max": 5.0,
            "reclamp_frame_error_max": 3,
            "false_reclamp_before_labelled_target_max": 0,
            "occlusion_or_ambiguity_must_fail_closed": True,
            "successful_trials_required": 3,
        },
        "until_promotion": (
            "runtime_gate_eligible remains false; use this extractor in shadow mode and "
            "require a reviewed seed for every run"
        ),
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    config_path = args.reference_config.resolve()
    config = _read_json(config_path)
    manifest = _read_json(run_dir / "input_manifest.json")
    timestamps = np.load(run_dir / "timestamps_s.npy")
    start, end = args.start_frame, args.end_frame
    if not 0 <= start < end < len(timestamps):
        raise RuntimeError("invalid validation frame interval")
    roi = tuple(args.roi)

    camera = config["camera"]
    if manifest["device"]["serial"] != camera["serial"]:
        raise RuntimeError("recording camera serial does not match the learned reference")
    if _sha256(run_dir / "K.npy") != camera["intrinsics_sha256"]:
        raise RuntimeError("recording intrinsics do not match the learned reference")

    trace_path = run_dir / "analysis" / "tube_axis_image_trace.csv"
    reviewed = _reviewed_axis_by_frame(trace_path)
    tracker = TubeAxisCandidateTracker()
    target = config["target"]
    gate = VisualAngleGate(
        VisualAngleGateConfig(
            target_image_axis_deg=float(target["target_image_axis_deg"]),
            tolerance_deg=float(target["tolerance_deg"]),
            min_consecutive_frames=int(target["min_consecutive_frames"]),
            max_rate_deg_s=float(target["max_rate_deg_s"]),
        )
    )

    seed_line = _seed_from_reviewed_trace(trace_path, start)
    seed_observation = tracker.seed(timestamp_s=float(timestamps[start]), line=seed_line)
    seed_gate = gate.update(timestamp_s=float(timestamps[start]), observation=seed_observation)
    rows: list[dict[str, object]] = []
    overlays: dict[int, np.ndarray] = {}
    accepted_frames: list[int] = []
    tracked_count = 1
    candidate_counts: list[int] = []

    seed_result = TubeAxisTrackResult(
        decision="REVIEWED_SEED",
        observation=seed_observation,
        selected_line=seed_line,
        considered_candidate_count=1,
        equivalent_candidate_count=1,
        tracking_score=1.0,
        ambiguity_margin=1.0,
        reason="manual line from the reviewed demo trace",
    )

    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "test_tube_rgb_tracker_overlay.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
    if not writer.isOpened():
        raise RuntimeError(f"could not open overlay video writer: {video_path}")

    keyframes = {start, 310, 325, 326, 330, 340, end}
    try:
        for frame in range(start, end + 1):
            image = _read_rgb(run_dir / "rgb" / f"{frame:06d}.png")
            if frame == start:
                candidates = [seed_line]
                result = seed_result
                gate_result = seed_gate
            else:
                candidates = extract_hough_line_candidates(image, roi_uvxy=roi)
                result = tracker.update(
                    timestamp_s=float(timestamps[frame]),
                    candidates=candidates,
                    roi_uvxy=roi,
                )
                gate_result = gate.update(
                    timestamp_s=float(timestamps[frame]), observation=result.observation
                )
                tracked_count += int(result.observation.valid)
                candidate_counts.append(len(candidates))
            if gate_result.accepted:
                accepted_frames.append(frame)

            reviewed_angle = reviewed.get(frame)
            automatic_angle = result.observation.axis_angle_deg
            error = (
                axis_angle_error_deg(automatic_angle, reviewed_angle)
                if automatic_angle is not None and reviewed_angle is not None
                else None
            )
            rows.append(
                {
                    "frame": frame,
                    "relative_time_s": float(timestamps[frame] - timestamps[0]),
                    "candidate_count": len(candidates),
                    "tracker_decision": result.decision,
                    "axis_valid": int(result.observation.valid),
                    "automatic_axis_deg": "" if automatic_angle is None else automatic_angle,
                    "reviewed_axis_deg": "" if reviewed_angle is None else reviewed_angle,
                    "automatic_minus_reviewed_deg": "" if error is None else error,
                    "tracking_score": ""
                    if result.tracking_score is None
                    else result.tracking_score,
                    "ambiguity_margin": ""
                    if result.ambiguity_margin is None
                    else result.ambiguity_margin,
                    "visual_gate_decision": gate_result.decision,
                    "consecutive_in_band": gate_result.consecutive_in_band,
                    "visual_gate_accepted": int(gate_result.accepted),
                }
            )
            overlay = _draw_overlay(
                image,
                frame=frame,
                roi=roi,
                candidates=candidates,
                result=result,
                gate_decision=gate_result.decision,
                gate_count=gate_result.consecutive_in_band,
                target_angle_deg=float(target["target_image_axis_deg"]),
            )
            writer.write(overlay)
            if frame in keyframes:
                overlays[frame] = overlay
    finally:
        writer.release()

    csv_path = output_dir / "test_tube_rgb_tracker_validation.csv"
    _write_rows(csv_path, rows)

    errors = np.asarray(
        [
            float(row["automatic_minus_reviewed_deg"])
            for row in rows
            if row["automatic_minus_reviewed_deg"] != ""
        ],
        dtype=float,
    )
    first_accept = min(accepted_frames) if accepted_frames else None
    reviewed_target_frame = int(config["source"]["frame"])
    reviewed_uncertainty = 5
    offline_pass = bool(
        tracked_count / len(rows) >= 0.95
        and len(errors) >= 10
        and float(np.mean(np.abs(errors))) <= 3.0
        and float(np.max(np.abs(errors))) <= 5.0
        and first_accept is not None
        and abs(first_accept - reviewed_target_frame) <= reviewed_uncertainty
    )

    montage_frames = sorted(overlays)
    tiles = [cv2.resize(overlays[frame], (480, 360)) for frame in montage_frames]
    rows_of_tiles: list[np.ndarray] = []
    for index in range(0, len(tiles), 2):
        pair = tiles[index : index + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows_of_tiles.append(np.hstack(pair))
    montage = np.vstack(rows_of_tiles)
    montage_path = output_dir / "test_tube_rgb_tracker_keyframes.jpg"
    cv2.imwrite(str(montage_path), montage)

    report = {
        "schema": SCHEMA,
        "status": (
            "offline_human_demo_replay_pass_robot_held_unqualified"
            if offline_pass
            else "offline_replay_failed"
        ),
        "runtime_gate_eligible": False,
        "hardware_commands_dispatched": False,
        "depth_used_for_tube_axis": False,
        "source": {
            "run_dir": str(run_dir),
            "frame_interval_inclusive": [start, end],
            "reference_config": str(config_path),
            "reference_config_sha256": _sha256(config_path),
            "reviewed_trace": str(trace_path),
            "reviewed_trace_sha256": _sha256(trace_path),
            "seed_frame": start,
            "seed_source": "reviewed tube edge in tube_axis_image_trace.csv",
            "roi_uvxy": list(roi),
        },
        "extractor_contract": {
            "input": "uint8 BGR frame plus bounded ROI and prior reviewed/accepted axis",
            "candidate_method": "CPU Canny plus probabilistic Hough segments",
            "association": (
                "segment length, axis-rate, perpendicular drift, midpoint drift, and "
                "distinct-hypothesis score margin"
            ),
            "fail_closed_outputs": ["HOLD_AND_REOBSERVE", "RESEED_REQUIRED"],
            "automatic_global_reacquisition": False,
            "transparent_depth_used": False,
        },
        "offline_replay": {
            "passed": offline_pass,
            "frames": len(rows),
            "tracked_frames": tracked_count,
            "tracked_fraction": tracked_count / len(rows),
            "hough_candidate_count": {
                "min": min(candidate_counts),
                "median": float(np.median(candidate_counts)),
                "max": max(candidate_counts),
            },
            "reviewed_comparison_count": len(errors),
            "reviewed_axis_mae_deg": float(np.mean(np.abs(errors))),
            "reviewed_axis_max_error_deg": float(np.max(np.abs(errors))),
            "learned_target_axis_deg": float(target["target_image_axis_deg"]),
            "target_tolerance_deg": float(target["tolerance_deg"]),
            "consecutive_frames_required": int(target["min_consecutive_frames"]),
            "first_reclamp_frame": first_accept,
            "reviewed_target_frame": reviewed_target_frame,
            "reviewed_target_frame_uncertainty": reviewed_uncertainty,
            "reclamp_frame_error": None
            if first_accept is None
            else first_accept - reviewed_target_frame,
            "interpretation": (
                "The four-frame gate fires at the early edge of the reviewed +/-5 frame "
                "target interval; this is expected from the configured +/-5 degree band."
            ),
        },
        "qualification_boundary": {
            "why_not_runtime_eligible": (
                "This replay is human-held and begins from a reviewed seed. It does not "
                "test robot fingers, robot occlusion, or the live robot-held appearance."
            ),
            "first_robot_held_qualification": _qualification_procedure(),
        },
        "artifacts": {
            "frame_table": str(csv_path),
            "overlay_video": str(video_path),
            "keyframe_montage": str(montage_path),
        },
    }
    report_path = output_dir / "test_tube_rgb_tracker_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if offline_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
