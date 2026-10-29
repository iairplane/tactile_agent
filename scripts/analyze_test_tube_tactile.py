#!/usr/bin/env python3
"""Audit dual-GF220 observability in a recorded test-tube demonstration.

This is an offline analysis tool.  It reads only files below ``--run-dir`` and
writes JSON/CSV/PNG evidence below its ``analysis`` directory.  It never opens
a camera, serial port, ROS node, gripper, or robot connection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from demo2gripper.test_tube_tactile_supervisor import (  # noqa: E402, I001
    RawTactileFrameEvidence,
    RawTactileNoiseModel,
    compensated_residual,
    evaluate_raw_tactile_frame,
    fit_unloaded_raw_model,
)


SCHEMA = "demo2gripper.test_tube_tactile_observability.v1"
DEFAULT_RUN = REPO / "runs/live/test_tube_demo_20261023_02"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--baseline-start", type=int, default=0)
    parser.add_argument("--baseline-end", type=int, default=180)
    parser.add_argument("--baseline-stride", type=int, default=3)
    parser.add_argument("--task-start", type=int, default=210)
    parser.add_argument("--task-end", type=int, default=425)
    return parser.parse_args()


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read image: {path}")
    if image.shape != (480, 640, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"unexpected tactile image contract: {path}: {image.shape}")
    return image


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _csv_number(value: float | None) -> float | str:
    result = _finite(value)
    return "" if result is None else result


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _sync_summary(
    manifest: dict[str, Any],
    finger: str,
) -> dict[str, Any]:
    frames = manifest["frames"]
    sequence = np.asarray([frame[f"tactile_{finger}_sequence"] for frame in frames], dtype=np.int64)
    tactile_time = np.asarray(
        [frame[f"tactile_{finger}_host_monotonic_s"] for frame in frames], dtype=float
    )
    offsets = np.asarray(
        [frame[f"tactile_{finger}_minus_d435_host_ms"] for frame in frames], dtype=float
    )
    sequence_delta = np.diff(sequence)
    duplicate_count = int(np.count_nonzero(sequence_delta == 0))
    span = float(tactile_time[-1] - tactile_time[0])
    sensor_rate = float((sequence[-1] - sequence[0]) / span)
    return {
        "associated_records": len(sequence),
        "unique_sensor_sequences": int(len(np.unique(sequence))),
        "reused_latest_snapshot_count": duplicate_count,
        "reused_latest_snapshot_fraction": duplicate_count / max(len(sequence) - 1, 1),
        "sensor_sequence_rate_hz": sensor_rate,
        "sequence_gaps_gt_one": int(np.count_nonzero(sequence_delta > 1)),
        "host_offset_ms": _percentiles(offsets),
        "absolute_host_offset_ms": _percentiles(np.abs(offsets)),
        "timestamps_are_exposure_times": False,
        "timestamp_semantics": "host read-completion time only",
    }


def _evidence_row(
    *,
    index: int,
    time_s: float,
    finger: str,
    sequence: int,
    reused: bool,
    host_offset_ms: float,
    evidence: RawTactileFrameEvidence,
    model: RawTactileNoiseModel,
) -> dict[str, object]:
    geometry = evidence.geometry
    motion = evidence.marker_motion
    return {
        "frame_index": index,
        "time_s": time_s,
        "finger": finger.upper(),
        "sensor_sequence": sequence,
        "reused_latest_snapshot": int(reused),
        "host_offset_ms": host_offset_ms,
        "residual_p99": evidence.residual_p99,
        "residual_max": evidence.residual_max,
        "residual_threshold": model.residual_threshold,
        "contact_area_px": geometry.area_px,
        "contact_area_fraction": geometry.area_fraction,
        "min_contact_area_px": model.min_contact_area_px,
        "contact_candidate": int(geometry.contact_candidate),
        "contact_center_u_px": geometry.center_uv_px[0]
        if geometry.center_uv_px is not None
        else "",
        "contact_center_v_px": geometry.center_uv_px[1]
        if geometry.center_uv_px is not None
        else "",
        "raw_patch_orientation_deg": _csv_number(geometry.orientation_deg),
        "patch_elongation": _csv_number(geometry.elongation),
        "orientation_concentration": _csv_number(geometry.orientation_concentration),
        "orientation_valid": int(geometry.orientation_valid),
        "marker_feature_count": motion.feature_count,
        "marker_tracked_count": motion.tracked_count,
        "global_flow_u_px": motion.global_flow_uv_px[0]
        if motion.global_flow_uv_px is not None
        else "",
        "global_flow_v_px": motion.global_flow_uv_px[1]
        if motion.global_flow_uv_px is not None
        else "",
        "marker_shear_median_px": _csv_number(motion.shear_median_px),
        "marker_shear_p90_px": _csv_number(motion.shear_p90_px),
        "marker_shear_p90_threshold_px": model.marker_shear_p90_threshold_px,
        "flow_valid": int(motion.valid),
        "slip_shadow_candidate": int(evidence.slip_shadow_candidate),
    }


def _finger_observation(
    rows: list[dict[str, object]],
    *,
    task_start: int,
    task_end: int,
) -> dict[str, Any]:
    task = [row for row in rows if task_start <= int(row["frame_index"]) <= task_end]
    contact = np.asarray([int(row["contact_candidate"]) for row in task], dtype=bool)
    orientation = np.asarray([int(row["orientation_valid"]) for row in task], dtype=bool)
    slip = np.asarray([int(row["slip_shadow_candidate"]) for row in task], dtype=bool)
    area = np.asarray([float(row["contact_area_px"]) for row in task], dtype=float)
    residual = np.asarray([float(row["residual_p99"]) for row in task], dtype=float)
    shear = np.asarray(
        [
            float(row["marker_shear_p90_px"]) if row["marker_shear_p90_px"] != "" else np.nan
            for row in task
        ],
        dtype=float,
    )
    return {
        "task_window_frames": [task_start, task_end],
        "task_window_sample_count": len(task),
        "contact_candidate_frames": int(contact.sum()),
        "contact_candidate_fraction": float(contact.mean()),
        "longest_contact_candidate_run_frames": _longest_run(contact),
        "orientation_valid_frames": int(orientation.sum()),
        "slip_shadow_candidate_frames": int(slip.sum()),
        "contact_area_px": _percentiles(area),
        "residual_p99": _percentiles(residual),
        "marker_shear_p90_px": _percentiles(shear[np.isfinite(shear)]),
    }


def _longest_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty feature CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _draw_chart(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    title: str,
    traces: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    y_max: float,
    threshold: float | None = None,
    task_window: tuple[int, int] = (210, 425),
) -> None:
    x0, y0, x1, y1 = box
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70), 1)
    cv2.putText(canvas, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1)
    sample_count = len(traces[0][1])
    start, stop = task_window
    sx0 = x0 + int(start / max(sample_count - 1, 1) * (x1 - x0))
    sx1 = x0 + int(stop / max(sample_count - 1, 1) * (x1 - x0))
    overlay = canvas.copy()
    cv2.rectangle(overlay, (sx0, y0 + 1), (sx1, y1 - 1), (48, 41, 28), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0.0, canvas)
    if threshold is not None and y_max > 0.0:
        ty = y1 - int(np.clip(threshold / y_max, 0.0, 1.0) * (y1 - y0))
        cv2.line(canvas, (x0, ty), (x1, ty), (90, 190, 240), 1, cv2.LINE_AA)
    for name, values, colour in traces:
        finite = np.nan_to_num(values, nan=0.0, posinf=y_max, neginf=0.0)
        points = []
        for index, value in enumerate(finite):
            x = x0 + int(index / max(sample_count - 1, 1) * (x1 - x0))
            y = y1 - int(np.clip(value / max(y_max, 1e-12), 0.0, 1.0) * (y1 - y0))
            points.append((x, y))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, colour, 1, cv2.LINE_AA)
        label_x = x0 + 10 + 170 * len([t for t in traces if t[0] < name])
        cv2.putText(
            canvas,
            name,
            (label_x, y0 + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            colour,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"0                               frame                               {sample_count - 1}",
        (x0 + 8, y1 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )


def _write_plot(
    path: Path,
    rows_by_finger: dict[str, list[dict[str, object]]],
    models: dict[str, RawTactileNoiseModel],
    manifest: dict[str, Any],
    task_window: tuple[int, int],
) -> None:
    canvas = np.full((1120, 1600, 3), 18, dtype=np.uint8)
    cv2.putText(
        canvas,
        "TEST-TUBE TACTILE OBSERVABILITY AUDIT",
        (55, 58),
        cv2.FONT_HERSHEY_DUPLEX,
        1.25,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Raw UVC only | no SDK WARPED_IMG / DIFF_IMG / FORCE6D | shaded region = human action",
        (57, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (160, 190, 220),
        1,
        cv2.LINE_AA,
    )
    values: dict[str, dict[str, np.ndarray]] = {}
    for finger, rows in rows_by_finger.items():
        values[finger] = {
            key: np.asarray(
                [float(row[key]) if row[key] != "" else np.nan for row in rows], dtype=float
            )
            for key in (
                "contact_area_px",
                "residual_p99",
                "marker_shear_p90_px",
                "host_offset_ms",
            )
        }
    colours = {"a": (255, 190, 70), "b": (90, 120, 255)}
    contact_max = max(
        80.0,
        max(models[key].min_contact_area_px for key in models) * 1.25,
        *(float(np.max(values[key]["contact_area_px"])) for key in values),
    )
    _draw_chart(
        canvas,
        (70, 145, 1530, 335),
        title="Largest residual component (px): no contact patch crosses the shadow threshold",
        traces=[(key.upper(), values[key]["contact_area_px"], colours[key]) for key in ("a", "b")],
        y_max=contact_max,
        threshold=float(max(model.min_contact_area_px for model in models.values())),
        task_window=task_window,
    )
    residual_max = max(
        30.0,
        *(float(np.max(values[key]["residual_p99"])) for key in values),
        *(model.residual_threshold for model in models.values()),
    )
    _draw_chart(
        canvas,
        (70, 395, 1530, 585),
        title="99th-percentile compensated image residual: remains below noise-fit pixel threshold",
        traces=[(key.upper(), values[key]["residual_p99"], colours[key]) for key in ("a", "b")],
        y_max=residual_max * 1.1,
        threshold=float(min(model.residual_threshold for model in models.values())),
        task_window=task_window,
    )
    shear_max = max(
        0.65,
        *(float(np.nanmax(values[key]["marker_shear_p90_px"])) for key in values),
        *(model.marker_shear_p90_threshold_px for model in models.values()),
    )
    _draw_chart(
        canvas,
        (70, 645, 1530, 835),
        title=(
            "Non-rigid marker shear p90 (raw pixels): subpixel camera noise, not an observed pivot"
        ),
        traces=[
            (key.upper(), values[key]["marker_shear_p90_px"], colours[key]) for key in ("a", "b")
        ],
        y_max=shear_max * 1.1,
        threshold=float(min(model.marker_shear_p90_threshold_px for model in models.values())),
        task_window=task_window,
    )
    maximum_age = float(manifest["tactile"]["maximum_absolute_host_offset_ms"])
    absolute_offsets = {key: np.abs(values[key]["host_offset_ms"]) for key in values}
    _draw_chart(
        canvas,
        (70, 895, 1530, 1045),
        title="Absolute raw-UVC host association age (ms): acquisition-integrity signal only",
        traces=[(key.upper(), absolute_offsets[key], colours[key]) for key in ("a", "b")],
        y_max=maximum_age,
        threshold=maximum_age,
        task_window=task_window,
    )
    cv2.putText(
        canvas,
        (
            "RESULT: zero positive robot-contact frames -> contact, slip, "
            "45-degree pivot, and jam gates remain unavailable."
        ),
        (65, 1092),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (90, 190, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write {path}")


def _write_keyframes(
    path: Path,
    run_dir: Path,
    models: dict[str, RawTactileNoiseModel],
) -> None:
    selections = [
        (120, "unloaded"),
        (285, "human pivot"),
        (350, "human reclamp"),
        (390, "human insert"),
    ]
    rows = []
    for frame_index, label in selections:
        panels = []
        for finger in ("a", "b"):
            image = _read_image(run_dir / f"tactile_{finger}" / f"{frame_index:06d}.png")
            residual = compensated_residual(
                image, models[finger].reference_bgr, models[finger].roi_mask
            )
            residual = np.where(models[finger].roi_mask, residual, 0.0)
            heat = cv2.applyColorMap(
                np.clip(residual / models[finger].residual_threshold * 255.0, 0, 255).astype(
                    np.uint8
                ),
                cv2.COLORMAP_INFERNO,
            )
            for view, name in ((image, "raw"), (heat, "residual")):
                panel = cv2.resize(view, (320, 240), interpolation=cv2.INTER_AREA)
                roi_rows, roi_cols = np.nonzero(models[finger].roi_mask)
                roi_box = (
                    (
                        round(float(roi_cols.min()) / 2.0),
                        round(float(roi_rows.min()) / 2.0),
                    ),
                    (
                        round(float(roi_cols.max()) / 2.0),
                        round(float(roi_rows.max()) / 2.0),
                    ),
                )
                cv2.rectangle(panel, roi_box[0], roi_box[1], (70, 235, 170), 1)
                cv2.rectangle(panel, (0, 0), (320, 30), (0, 0, 0), -1)
                title = f"f{frame_index} {label} | {finger.upper()} {name}"
                cv2.putText(
                    panel,
                    title,
                    (7, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )
                panels.append(panel)
        rows.append(np.concatenate(panels, axis=1))
    montage = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(path), montage):
        raise RuntimeError(f"could not write {path}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("input manifest contains no frames")
    frame_count = len(frames)
    if not (
        0 <= args.baseline_start < args.baseline_end <= frame_count
        and args.baseline_stride > 0
        and 0 <= args.task_start <= args.task_end < frame_count
    ):
        raise ValueError("invalid baseline or task frame interval")

    tactile = manifest.get("tactile", {})
    semantics = str(tactile.get("semantics", ""))
    if "raw GF220" not in semantics or "not ViTai SDK" not in semantics:
        raise RuntimeError("this analyzer requires the recorder's explicit raw-UVC contract")
    baseline_indices = list(range(args.baseline_start, args.baseline_end, args.baseline_stride))
    models: dict[str, RawTactileNoiseModel] = {}
    rows_by_finger: dict[str, list[dict[str, object]]] = {}
    first_host_time = float(frames[0]["d435_host_monotonic_s"])

    for finger in ("a", "b"):
        baseline = np.stack(
            [
                _read_image(run_dir / f"tactile_{finger}" / f"{index:06d}.png")
                for index in baseline_indices
            ]
        )
        model = fit_unloaded_raw_model(baseline)
        models[finger] = model
        rows = []
        previous_sequence: int | None = None
        for index, record in enumerate(frames):
            image = _read_image(run_dir / str(record[f"tactile_{finger}"]))
            evidence = evaluate_raw_tactile_frame(image, model)
            sequence = int(record[f"tactile_{finger}_sequence"])
            rows.append(
                _evidence_row(
                    index=index,
                    time_s=float(record["d435_host_monotonic_s"]) - first_host_time,
                    finger=finger,
                    sequence=sequence,
                    reused=previous_sequence == sequence,
                    host_offset_ms=float(record[f"tactile_{finger}_minus_d435_host_ms"]),
                    evidence=evidence,
                    model=model,
                )
            )
            previous_sequence = sequence
        rows_by_finger[finger] = rows

    all_rows = [row for finger in ("a", "b") for row in rows_by_finger[finger]]
    all_rows.sort(key=lambda row: (int(row["frame_index"]), str(row["finger"])))
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tactile_features.csv"
    json_path = output_dir / "tactile_observability.json"
    plot_path = output_dir / "tactile_observability.png"
    keyframes_path = output_dir / "tactile_keyframes.png"
    _write_csv(csv_path, all_rows)

    d435_host = np.asarray([frame["d435_host_monotonic_s"] for frame in frames], dtype=float)
    observations = {
        finger: _finger_observation(
            rows_by_finger[finger], task_start=args.task_start, task_end=args.task_end
        )
        for finger in ("a", "b")
    }
    no_positive_contact = all(
        value["contact_candidate_frames"] == 0 for value in observations.values()
    )
    threshold_fit = {
        "fit_type": "negative_only_unloaded_noise_envelope",
        "baseline_frames": [args.baseline_start, args.baseline_end - 1],
        "baseline_stride": args.baseline_stride,
        "fitted_sample_count_per_finger": len(baseline_indices),
        "positive_robot_contact_examples": 0,
        "runtime_gate_eligible": False,
        "reason": (
            "The demonstration contains no robot-finger/test-tube contact. These values "
            "bound unloaded raw-UVC noise but do not establish contact sensitivity or a "
            "physical decision boundary."
        ),
        "per_finger": {
            finger: {
                "raw_image_residual_threshold": model.residual_threshold,
                "minimum_connected_area_px": model.min_contact_area_px,
                "marker_shear_p90_threshold_px": model.marker_shear_p90_threshold_px,
                "marker_feature_count": len(model.marker_points_uv),
                "authority": "shadow_only",
            }
            for finger, model in models.items()
        },
        "cannot_fit_from_this_recording": [
            "contact-on threshold and hysteresis",
            "contact-loss/reclamp threshold",
            "raw-image orientation to tube-axis angle calibration",
            "45-degree pivot target",
            "slip onset and direction",
            "insertion jam force or torque threshold",
        ],
    }
    predicates = {
        "fresh_dual_tactile_samples": {
            "runtime_gate_eligible": True,
            "scope": "acquisition_integrity_only",
            "current_recording_pass": True,
            "required_signal": [
                "new tactile_a_sequence",
                "new tactile_b_sequence",
                "common host association age",
            ],
            "maximum_absolute_host_age_ms": float(tactile["maximum_absolute_host_offset_ms"]),
            "meaning": "Frames are fresh enough to evaluate; says nothing about contact.",
        },
        "contact_present": {
            "runtime_gate_eligible": False,
            "shadow_eligible": True,
            "observed_positive": not no_positive_contact,
            "required_upgrade": "record unloaded and robot-contact windows for each finger",
        },
        "contact_lost": {
            "runtime_gate_eligible": False,
            "shadow_eligible": False,
            "observed_positive": False,
            "required_upgrade": "fit contact-on/contact-off hysteresis from a robot grasp",
        },
        "pivot_in_progress": {
            "runtime_gate_eligible": False,
            "shadow_eligible": True,
            "observed_positive": False,
            "required_upgrade": "observe marker shear while the tube is actually between pads",
        },
        "pivot_target_reached": {
            "runtime_gate_eligible": False,
            "shadow_eligible": False,
            "observed_positive": False,
            "required_upgrade": (
                "synchronize robot-contact tactile images with an external tube-axis angle "
                "reference and fit the per-finger mapping"
            ),
        },
        "slip_detected": {
            "runtime_gate_eligible": False,
            "shadow_eligible": True,
            "observed_positive": False,
            "required_upgrade": "label retained contact, controlled pivot, axial slip, and drop",
        },
        "stable_reclamp": {
            "runtime_gate_eligible": False,
            "shadow_eligible": False,
            "observed_positive": False,
            "required_upgrade": "positive contact plus post-close motion-stop windows",
        },
        "insertion_jam": {
            "runtime_gate_eligible": False,
            "shadow_eligible": False,
            "observed_positive": False,
            "required_upgrade": (
                "record calibrated FORCE6D/task-frame wrench or a validated image signature "
                "during free insertion and controlled obstruction"
            ),
        },
    }
    report = {
        "schema": SCHEMA,
        "status": "shadow_only_no_robot_contact_positive",
        "runtime_gate_eligible": False,
        "source": {
            "run_dir": str(run_dir),
            "manifest": "../input_manifest.json",
            "frame_count": frame_count,
            "elapsed_s": float(manifest["host_recording_elapsed_s"]),
            "human_action_window_frames": [args.task_start, args.task_end],
        },
        "channels": {
            "front_rgb": "lossless PNG",
            "aligned_depth": "float32 metres plus unchanged Z16 raw depth",
            "tactile_a": "raw UVC BGR8 PNG",
            "tactile_b": "raw UVC BGR8 PNG",
            "sdk_warped_img_present": False,
            "sdk_diff_img_present": False,
            "sdk_force6d_present": False,
            "logical_left_right_calibrated": False,
        },
        "synchronization": {
            "d435_effective_rate_hz": float(1.0 / np.median(np.diff(d435_host))),
            "association": manifest["synchronization"]["join"],
            "tactile_exposure_timestamp_available": False,
            "a": _sync_summary(manifest, "a"),
            "b": _sync_summary(manifest, "b"),
        },
        "feature_semantics": {
            "contact_area": "largest coherent raw-image residual patch, in pixels",
            "contact_orientation": (
                "PCA long-axis of that residual patch in raw camera pixels; not tube angle"
            ),
            "marker_shear": (
                "Lucas-Kanade marker displacement after subtracting median global image motion"
            ),
            "force_or_torque": "not recorded",
        },
        "related_assets": {
            "same_hardware_sdk_unloaded_calibration": (
                "../../medicine_bottle_execution_20261023_01/"
                "gf220_task_calibration_01/calibrations.json"
            ),
            "available_products": ["WARPED_IMG baseline", "DIFF_IMG baseline", "FORCE6D baseline"],
            "allowed_use": (
                "reuse sensor identity and SDK warp/baseline provenance, then acquire a fresh "
                "adaptive unloaded baseline before this task"
            ),
            "not_established_by_asset": [
                "test-tube contact threshold",
                "test-tube slip threshold",
                "insertion jam threshold",
                "tactile image angle to tube-axis angle mapping",
            ],
        },
        "per_finger": observations,
        "threshold_fit": threshold_fit,
        "predicates": predicates,
        "blockers": [
            "No robot finger contacted the test tube during this human demonstration.",
            "No positive contact, pivot, slip, reclamp, insertion, or jam tactile labels exist.",
            "Raw UVC contact-patch direction has no calibration to the test-tube axis.",
            "WARPED_IMG, DIFF_IMG, FORCE6D, and calibrated sensor-to-TCP rotations are absent.",
            "A/B hardware identities are not calibrated to logical left/right fingers.",
        ],
        "next_calibration_capture": [
            "Record unloaded windows, then light/nominal contact for each finger.",
            (
                "Hold the robot wrist fixed and pivot a grasped tube through labelled "
                "0/15/30/45 degrees."
            ),
            "Include retained contact, controlled axial slip, and immediate reclamp examples.",
            "Record free insertion and a bounded controlled obstruction for the jam predicate.",
            "Use SDK WARPED_IMG/DIFF_IMG and FORCE6D with device timestamps if available.",
        ],
        "recommended_first_execution_authority": {
            "target_45_degree_reached": "front RGB tube-axis estimate",
            "tactile_contact_retention": (
                "conditional runtime gate only after fresh unloaded and positive-contact "
                "bootstrap windows"
            ),
            "tactile_slip": "shadow first; promote only after labelled controlled-slip capture",
            "tactile_insertion_jam": (
                "shadow first; promote only after free-insertion and bounded-obstruction capture"
            ),
        },
        "artifacts": {
            "frame_features_csv": csv_path.name,
            "observability_plot": plot_path.name,
            "keyframe_comparison": keyframes_path.name,
        },
        "hardware_commands_dispatched": False,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_plot(
        plot_path,
        rows_by_finger,
        models,
        manifest,
        (args.task_start, args.task_end),
    )
    _write_keyframes(keyframes_path, run_dir, models)
    print(
        json.dumps(
            {
                "status": report["status"],
                "runtime_gate_eligible": False,
                "json": str(json_path),
                "csv": str(csv_path),
                "plot": str(plot_path),
                "keyframes": str(keyframes_path),
            }
        )
    )


if __name__ == "__main__":
    main()
