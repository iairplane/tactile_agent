#!/usr/bin/env python3
"""Render the nested-cup demo from one finalized front + tactile capture.

This is an offline, read-only renderer.  It never opens a camera, gripper, or
robot connection.  The tactile panels are perspective-warped from the saved raw
UVC frames and the component counts are local baseline-``absdiff`` diagnostics.
They are not ViTai SDK ``WARPED_IMG``/``DIFF_IMG`` or a calibrated classifier.

The action log supplies the story phase.  In particular, ``LOW PROBE / RETRY``
describes the scripted low probe; it is not presented as a tactile rejection.
An action-log phase or a pose/width check cannot create a pass by itself.
``DEMO GATE PASS`` requires a stable observed 1/2-region template, unless an
operator outcome explicitly confirms that exactly one cup was picked.  Measured
diagnostic component counts remain visible and are saved for every video frame.

An optional operator-outcome file has final authority over the recorded task
result.  If the operator reports ``multi_cup_pick``, lift/hold frames are marked
``OPERATOR-VERIFIED: TWO CUPS / FAIL`` and can never show a pass badge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SCHEMA = "demo2gripper.cup_tactile_demo_render.v1"
CAPTURE_SCHEMA = "demo2gripper.realsense_rgbd_raw_tactile.v1"
OPERATOR_OUTCOME_SCHEMA = "demo2gripper.cup_pick_operator_outcome.v1"
DEFAULT_WARP_PROFILE = (
    Path(__file__).resolve().parents[1] / "configs/gf220_manual_warp_draft.json"
)

SERIAL_A = "184603"
SERIAL_B = "196FB3"
EXPECTED_COMPONENTS_A = 1
EXPECTED_COMPONENTS_B = 2

PHASE_READY = "READY"
PHASE_LOW = "LOW_PROBE"
PHASE_RETRY = "RETRY"
PHASE_REVIEWED_CLOSE = "REVIEWED_CLOSE"
PHASE_SUCCESS = "SUCCESS_GRASP"
PHASE_LIFT = "LIFT"
PHASE_HOLD = "HOLD"
PHASE_DONE = "DONE"
PHASES = (
    PHASE_READY,
    PHASE_LOW,
    PHASE_RETRY,
    PHASE_REVIEWED_CLOSE,
    PHASE_SUCCESS,
    PHASE_LIFT,
    PHASE_HOLD,
    PHASE_DONE,
)


@dataclass(frozen=True)
class PhaseEvent:
    monotonic_s: float
    phase: str
    source: str


@dataclass(frozen=True)
class WarpSpec:
    serial_short: str
    corners_xy: np.ndarray
    output_size: tuple[int, int]


@dataclass(frozen=True)
class TactileMetric:
    component_count: int
    active_area_px: int
    diff_mean: float
    diff_p95: float
    threshold: float
    mask: np.ndarray
    diff_u8: np.ndarray


@dataclass(frozen=True)
class OperatorOutcome:
    outcome: str
    observed_cup_count: int | None
    source: str
    path: Path

    @property
    def is_multi_cup_pick(self) -> bool:
        return self.outcome == "multi_cup_pick"


@dataclass(frozen=True)
class VideoBadge:
    label: str
    color: tuple[int, int, int]
    detail: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--action-log", type=Path, required=True)
    parser.add_argument(
        "--operator-outcome",
        type=Path,
        default=None,
        help="Optional operator-verified task outcome JSON; overrides lift/hold badges",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warp-profile", type=Path, default=DEFAULT_WARP_PROFILE)
    parser.add_argument("--video-name", default="cup_tactile_demo.mp4")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--diff-threshold", type=float, default=16.0)
    parser.add_argument("--min-component-area-px", type=int, default=120)
    parser.add_argument("--baseline-max-frames", type=int, default=45)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--stable-match-frames", type=int, default=3)
    return parser.parse_args()


def _positive_number(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.fps is not None:
        _positive_number(args.fps, "--fps")
    _positive_number(args.diff_threshold, "--diff-threshold")
    for name in (
        "min_component_area_px",
        "baseline_max_frames",
        "temporal_window",
        "stable_match_frames",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be a positive integer")
    if args.temporal_window % 2 == 0:
        raise ValueError("--temporal-window must be odd")
    if Path(args.video_name).name != args.video_name or not args.video_name.endswith(".mp4"):
        raise ValueError("--video-name must be a plain .mp4 filename")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_operator_outcome(path: Path | None) -> OperatorOutcome | None:
    if path is None:
        return None
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("operator outcome must be a JSON object")
    schema = payload.get("schema")
    if schema != OPERATOR_OUTCOME_SCHEMA:
        raise ValueError(
            f"operator outcome schema must be {OPERATOR_OUTCOME_SCHEMA!r}; got {schema!r}"
        )
    outcome = payload.get("outcome")
    if outcome not in {"single_cup_pick", "multi_cup_pick", "no_pick", "unknown"}:
        raise ValueError("operator outcome is not a supported cup-pick result")
    observed_value = payload.get("observed_cup_count")
    lifted_value = payload.get("lifted_cup_count")
    if observed_value is not None and lifted_value is not None and observed_value != lifted_value:
        raise ValueError(
            "operator observed_cup_count and lifted_cup_count disagree"
        )
    count_value = observed_value if observed_value is not None else lifted_value
    if count_value is None:
        count = None
    elif isinstance(count_value, bool) or not isinstance(count_value, int) or count_value < 0:
        raise ValueError(
            "operator observed_cup_count/lifted_cup_count must be a non-negative integer or null"
        )
    else:
        count = count_value
    if outcome == "multi_cup_pick" and (count is None or count < 2):
        raise ValueError("multi_cup_pick requires observed_cup_count >= 2")
    if outcome == "single_cup_pick" and count != 1:
        raise ValueError("single_cup_pick requires an observed/lifted cup count of 1")
    if outcome == "no_pick" and count not in (None, 0):
        raise ValueError("no_pick requires an observed/lifted cup count of 0 or null")
    source = payload.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("operator outcome source must be a non-empty string")
    return OperatorOutcome(outcome, count, source.strip(), path.resolve())


def load_capture(capture_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    manifest_path = capture_dir / "input_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"{manifest_path} is not a {CAPTURE_SCHEMA} manifest")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("capture manifest contains no frames")
    times = np.asarray(
        [frame.get("d435_host_monotonic_s") for frame in frames], dtype=np.float64
    )
    if times.shape != (len(frames),) or not np.isfinite(times).all():
        raise ValueError("capture has invalid D435 host monotonic timestamps")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("capture host monotonic timestamps are not strictly increasing")
    required = ("rgb", "tactile_a", "tactile_b")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"capture frame {index} is not an object")
        for key in required:
            relative = frame.get(key)
            if not isinstance(relative, str) or not (capture_dir / relative).is_file():
                raise ValueError(f"capture frame {index} is missing {key!r}")
    return manifest, frames, times


def _short_serial(value: object) -> str:
    text = str(value).upper()
    if "184603" in text:
        return SERIAL_A
    if "196FB3" in text:
        return SERIAL_B
    raise ValueError(f"Unsupported tactile serial in warp profile: {value!r}")


def load_warp_specs(path: Path) -> dict[str, WarpSpec]:
    payload = _read_json(path)
    output = payload.get("output") if isinstance(payload, dict) else None
    sensors = payload.get("sensors") if isinstance(payload, dict) else None
    if not isinstance(output, dict) or not isinstance(sensors, list):
        raise ValueError("warp profile must contain output and sensors")
    width = int(output.get("width", 0))
    height = int(output.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("warp profile output dimensions must be positive")
    specs: dict[str, WarpSpec] = {}
    for item in sensors:
        if not isinstance(item, dict):
            raise ValueError("warp profile sensor entry must be an object")
        serial = _short_serial(item.get("sdk_serial"))
        corners = np.asarray(item.get("corners_xy"), dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError(f"sensor {serial} needs four finite warp corners")
        specs[serial] = WarpSpec(serial, corners, (width, height))
    if set(specs) != {SERIAL_A, SERIAL_B}:
        raise ValueError("warp profile must contain serials 184603 and 196FB3")
    return specs


def warp_raw(frame: np.ndarray, spec: WarpSpec) -> np.ndarray:
    width, height = spec.output_size
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(spec.corners_xy, destination)
    return cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _flatten_log_entries(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("action log must be an object or a list of objects")
    for key in ("events", "actions", "stages", "records", "entries"):
        value = payload.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, Mapping)]
    if all(isinstance(value, Mapping) for value in payload.values()):
        return [value for value in payload.values() if isinstance(value, Mapping)]
    raise ValueError("action log has no events/actions/stages/records list")


def _event_time(entry: Mapping[str, Any], capture_start_s: float) -> float | None:
    for key in (
        "host_monotonic_s",
        "monotonic_s",
        "timestamp_monotonic_s",
        "monotonic_time_s",
        "time_monotonic_s",
        "t_monotonic_s",
        "started_host_monotonic_s",
        "start_host_monotonic_s",
        "start_monotonic_s",
    ):
        value = entry.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    for key in ("elapsed_s", "relative_s", "recording_elapsed_s"):
        value = entry.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return capture_start_s + float(value)
    return None


def _entry_text(entry: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "video_phase",
        "phase",
        "stage",
        "action",
        "event",
        "name",
        "label",
        "status",
        "command",
        "step",
        "type",
        "state",
        "description",
        "message",
    ):
        value = entry.get(key)
        if isinstance(value, str):
            values.append(value)
    return " ".join(values).strip()


def _canonical_phase(entry: Mapping[str, Any]) -> str | None:
    name = entry.get("name")
    if isinstance(name, str):
        normalized_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        exact_names = {
            "recording_start": PHASE_READY,
            "recording_end": PHASE_DONE,
            "low_probe_descend_3mm": PHASE_LOW,
            "low_probe_descend": PHASE_LOW,
            "low_probe_close": PHASE_LOW,
            "low_probe_shadow_gate": PHASE_LOW,
            # The unload belongs to recovery even though its name retains the
            # low-probe prefix.
            "low_probe_unload": PHASE_RETRY,
            "return_to_reviewed_success_pose": PHASE_RETRY,
            "reviewed_pose_live_gate": PHASE_RETRY,
            "reviewed_pose_close": PHASE_REVIEWED_CLOSE,
            "reviewed_pose_opening_gate_readback": PHASE_REVIEWED_CLOSE,
            "reviewed_pose_opening_gate": PHASE_REVIEWED_CLOSE,
            "reviewed_pose_shadow_gate": PHASE_SUCCESS,
            "vertical_lift": PHASE_LIFT,
            "cup_held_at_lift": PHASE_HOLD,
        }
        if normalized_name in exact_names:
            return exact_names[normalized_name]

    for key in ("video_phase", "phase"):
        value = entry.get(key)
        if isinstance(value, str):
            direct = re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
            aliases = {
                "BASELINE": PHASE_READY,
                "OPEN_BASELINE": PHASE_READY,
                "READY": PHASE_READY,
                "LOW": PHASE_LOW,
                "LOW_PROBE": PHASE_LOW,
                "LOW_PROBE_CLOSE": PHASE_LOW,
                "RETRY": PHASE_RETRY,
                "UNLOAD": PHASE_RETRY,
                "RETURN": PHASE_RETRY,
                "RETURN_REVIEWED": PHASE_RETRY,
                "SUCCESS": PHASE_SUCCESS,
                "SUCCESS_GRASP": PHASE_SUCCESS,
                "REVIEWED_GRASP": PHASE_SUCCESS,
                "FINAL_GRASP": PHASE_SUCCESS,
                "LIFT": PHASE_LIFT,
                "HOLD": PHASE_HOLD,
                "DONE": PHASE_DONE,
            }
            if direct in aliases:
                return aliases[direct]

    text = _entry_text(entry).lower()
    if not text:
        return None
    # Specific multiword stages take priority over generic words such as close/open.
    if any(token in text for token in ("low_probe", "low probe", "lower_probe", "too_deep")):
        return PHASE_LOW
    if any(
        token in text
        for token in ("return_reviewed", "return reviewed", "return_success", "unload", "retry")
    ):
        return PHASE_RETRY
    if any(
        token in text
        for token in (
            "success_grasp",
            "successful_grasp",
            "reviewed_grasp",
            "reviewed grasp",
            "final_grasp",
            "good_grasp",
            "nominal_grasp",
        )
    ):
        return PHASE_SUCCESS
    if any(token in text for token in ("lift", "raise_cup", "raise cup")):
        return PHASE_LIFT
    if "hold" in text:
        return PHASE_HOLD
    if any(
        token in text
        for token in ("recording_done", "recording_end", "recording end", "complete", "finished")
    ):
        return PHASE_DONE
    if any(
        token in text
        for token in ("baseline", "recording_start", "recording start", "ready", "open_initial")
    ):
        return PHASE_READY
    return None


def load_phase_events(path: Path, capture_start_s: float) -> list[PhaseEvent]:
    payload = _read_json(path)
    entries = _flatten_log_entries(payload)
    events: list[PhaseEvent] = []
    for entry in entries:
        phase = _canonical_phase(entry)
        monotonic_s = _event_time(entry, capture_start_s)
        if phase is None or monotonic_s is None:
            continue
        events.append(PhaseEvent(monotonic_s, phase, _entry_text(entry)))
    events.sort(key=lambda event: event.monotonic_s)
    if not events:
        raise ValueError(
            "action log contains no phase events with host monotonic timestamps; "
            "write video_phase plus host_monotonic_s for each scripted phase"
        )
    if events[0].monotonic_s > capture_start_s:
        events.insert(0, PhaseEvent(capture_start_s, PHASE_READY, "implicit capture start"))
    return events


def phases_for_frames(times: np.ndarray, events: Sequence[PhaseEvent]) -> list[str]:
    event_times = np.asarray([event.monotonic_s for event in events], dtype=np.float64)
    phases: list[str] = []
    for timestamp in times:
        index = int(np.searchsorted(event_times, timestamp, side="right") - 1)
        phases.append(events[max(index, 0)].phase)
    return phases


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def baseline_indices(phases: Sequence[str], maximum: int) -> list[int]:
    indices: list[int] = []
    for index, phase in enumerate(phases):
        if phase != PHASE_READY:
            break
        indices.append(index)
    if not indices:
        raise ValueError("capture has no READY/baseline frames before the low probe")
    return indices[-maximum:]


def make_baseline(
    capture_dir: Path,
    frames: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    key: str,
    spec: WarpSpec,
) -> np.ndarray:
    images = [
        warp_raw(_load_bgr(capture_dir / str(frames[index][key])), spec)
        for index in indices
    ]
    return np.median(np.stack(images, axis=0), axis=0).astype(np.uint8)


def _difference_magnitude(warped: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    # Lab reduces sensitivity to raw-camera channel balance while retaining both
    # luminance and colour changes produced by gel deformation.
    current_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    baseline_lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB).astype(np.float32)
    magnitude = np.linalg.norm(current_lab - baseline_lab, axis=2)
    return cv2.GaussianBlur(magnitude, (5, 5), 0.0)


def tactile_metric(
    magnitude: np.ndarray,
    *,
    threshold: float,
    min_component_area_px: int,
) -> TactileMetric:
    raw_mask = (magnitude >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    keep = np.zeros_like(cleaned)
    areas: list[int] = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area >= min_component_area_px:
            keep[labels == component] = 255
            areas.append(area)
    scaled = np.clip(magnitude * (255.0 / max(3.0 * threshold, 1.0)), 0, 255).astype(np.uint8)
    return TactileMetric(
        component_count=len(areas),
        active_area_px=int(sum(areas)),
        diff_mean=float(np.mean(magnitude)),
        diff_p95=float(np.percentile(magnitude, 95.0)),
        threshold=float(threshold),
        mask=keep,
        diff_u8=scaled,
    )


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(
        image,
        (max(1, round(source_w * scale)), max(1, round(source_h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _text(
    canvas: np.ndarray,
    value: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.8,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 2,
) -> None:
    cv2.putText(
        canvas,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        value,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _phase_title(phase: str) -> tuple[str, tuple[int, int, int]]:
    return {
        PHASE_READY: ("READY / UNLOADED BASELINE", (230, 210, 80)),
        PHASE_LOW: ("LOW PROBE / RETRY", (40, 170, 255)),
        PHASE_RETRY: ("UNLOAD -> RETURN TO REVIEWED Z", (40, 210, 255)),
        PHASE_REVIEWED_CLOSE: ("REVIEWED HEIGHT / GENTLE CLOSE", (80, 210, 230)),
        PHASE_SUCCESS: ("REVIEWED GRASP / TACTILE TEMPLATE", (80, 230, 120)),
        PHASE_LIFT: ("LIFT / CONTACT MAINTAINED", (80, 230, 120)),
        PHASE_HOLD: ("HOLD", (80, 230, 120)),
        PHASE_DONE: ("DEMO COMPLETE", (220, 220, 220)),
    }.get(phase, (phase, (230, 230, 230)))


def _tactile_panel(
    warped: np.ndarray,
    metric: TactileMetric,
    *,
    serial: str,
    expected: int,
    panel_size: tuple[int, int],
) -> np.ndarray:
    width, height = panel_size
    annotated = warped.copy()
    contours, _ = cv2.findContours(metric.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (70, 240, 120), 2, cv2.LINE_AA)
    panel = _fit(annotated, width, height)
    heat = cv2.applyColorMap(metric.diff_u8, cv2.COLORMAP_TURBO)
    heat = cv2.resize(heat, (118, 118), interpolation=cv2.INTER_AREA)
    panel[height - 128 : height - 10, width - 128 : width - 10] = heat
    cv2.rectangle(panel, (width - 129, height - 129), (width - 9, height - 9), (230, 230, 230), 1)
    _text(panel, f"SERIAL {serial}: {expected}-region template", (14, 31), scale=0.57)
    matched = metric.component_count == expected
    _text(
        panel,
        f"observed CC: {metric.component_count}   area: {metric.active_area_px}px",
        (14, height - 18),
        scale=0.50,
        color=(80, 230, 120) if matched else (60, 190, 255),
        thickness=1,
    )
    return panel


def badge_for_frame(
    phase: str,
    *,
    pair_match: bool,
    shadow_gate_latched: bool,
    operator_outcome: OperatorOutcome | None,
) -> VideoBadge:
    if (
        operator_outcome is not None
        and operator_outcome.is_multi_cup_pick
        and phase in {PHASE_LIFT, PHASE_HOLD, PHASE_DONE}
    ):
        return VideoBadge(
            "OPERATOR-VERIFIED: TWO CUPS / FAIL",
            (45, 45, 245),
            "lift outcome overrides the offline shadow template",
            "operator_outcome",
        )
    if (
        operator_outcome is not None
        and operator_outcome.outcome == "no_pick"
        and phase in {PHASE_LIFT, PHASE_HOLD, PHASE_DONE}
    ):
        return VideoBadge(
            "OPERATOR-VERIFIED: NO CUP / FAIL",
            (45, 45, 245),
            "lift outcome overrides the offline shadow template",
            "operator_outcome",
        )
    if phase == PHASE_LOW:
        return VideoBadge(
            "LOW PROBE / RETRY",
            (40, 170, 255),
            "scripted pose trial / no tactile rejection claim",
            "action_log_phase",
        )
    if phase == PHASE_RETRY:
        return VideoBadge(
            "RESET CONTACT",
            (40, 210, 255),
            "unload -> reviewed successful height",
            "action_log_phase",
        )
    if phase == PHASE_REVIEWED_CLOSE:
        if shadow_gate_latched:
            return VideoBadge(
                "TACTILE SHADOW STABLE MATCH",
                (70, 220, 110),
                "stable offline 1/2-region diagnostic",
                "raw_uvc_shadow_diagnostic",
            )
        if pair_match:
            return VideoBadge(
                "TACTILE SHADOW MATCH / STABILIZING",
                (80, 210, 230),
                "current frame matches; stable window not reached",
                "raw_uvc_shadow_diagnostic",
            )
        return VideoBadge(
            "SHADOW GATE NOT MATCHED",
            (40, 120, 255),
            "observed components do not match the 1/2-region template",
            "raw_uvc_shadow_diagnostic",
        )
    if phase in {PHASE_SUCCESS, PHASE_LIFT, PHASE_HOLD, PHASE_DONE}:
        if operator_outcome is not None and operator_outcome.outcome == "single_cup_pick":
            return VideoBadge(
                "OPERATOR-VERIFIED: ONE CUP / PASS",
                (70, 220, 110),
                "operator task outcome overrides the offline shadow template",
                "operator_outcome",
            )
        if operator_outcome is not None and operator_outcome.is_multi_cup_pick:
            label = (
                "SHADOW GATE FALSE POSITIVE"
                if shadow_gate_latched
                else "TACTILE SHADOW NOT MATCHED"
            )
            return VideoBadge(
                label,
                (40, 120, 255),
                "operator later verified a two-cup task failure",
                "operator_outcome",
            )
        if shadow_gate_latched:
            return VideoBadge(
                "DEMO GATE PASS",
                (70, 220, 110),
                "stable observed 1/2-region tactile shadow",
                "raw_uvc_shadow_diagnostic",
            )
        return VideoBadge(
            "POSE+WIDTH PASS / TACTILE SHADOW NOT MATCHED",
            (40, 170, 255),
            "motion continued by scripted demo; task outcome unverified",
            "action_log_plus_raw_uvc_shadow",
        )
    return VideoBadge(
        "TACTILE DIAGNOSTIC",
        (210, 210, 210),
        "waiting for scripted contact stage",
        "action_log_phase",
    )


def compose_frame(
    rgb_bgr: np.ndarray,
    warped_a: np.ndarray,
    warped_b: np.ndarray,
    metric_a: TactileMetric,
    metric_b: TactileMetric,
    *,
    phase: str,
    elapsed_s: float,
    pair_match: bool,
    stable_count: int,
    stable_required: int,
    shadow_gate_latched: bool,
    operator_outcome: OperatorOutcome | None,
) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), 12, dtype=np.uint8)
    title, phase_color = _phase_title(phase)
    _text(canvas, title, (34, 58), scale=1.22, color=phase_color, thickness=3)
    _text(canvas, f"t = {elapsed_s:05.1f}s", (1690, 54), scale=0.72, color=(190, 190, 190))

    front = _fit(rgb_bgr, 960, 720)
    canvas[112:832, 32:992] = front
    _text(canvas, "FRONT RGB", (48, 148), scale=0.72)

    panel_a = _tactile_panel(
        warped_a, metric_a, serial=SERIAL_A, expected=EXPECTED_COMPONENTS_A, panel_size=(400, 400)
    )
    panel_b = _tactile_panel(
        warped_b, metric_b, serial=SERIAL_B, expected=EXPECTED_COMPONENTS_B, panel_size=(400, 400)
    )
    canvas[112:512, 1040:1440] = panel_a
    canvas[112:512, 1480:1880] = panel_b

    badge = badge_for_frame(
        phase,
        pair_match=pair_match,
        shadow_gate_latched=shadow_gate_latched,
        operator_outcome=operator_outcome,
    )

    cv2.rectangle(canvas, (1040, 558), (1880, 662), (26, 26, 26), -1)
    cv2.rectangle(canvas, (1040, 558), (1880, 662), badge.color, 3)
    badge_scale = 0.83 if len(badge.label) > 28 else 1.12
    _text(canvas, badge.label, (1070, 613), scale=badge_scale, color=badge.color, thickness=3)
    _text(canvas, badge.detail, (1072, 648), scale=0.52, color=(200, 200, 200), thickness=1)

    match_color = (80, 230, 120) if pair_match else (50, 190, 255)
    _text(
        canvas,
        f"observed template pair: {'MATCH' if pair_match else 'not matched'}",
        (1050, 724),
        scale=0.72,
        color=match_color,
    )
    _text(
        canvas,
        f"temporal stability: {min(stable_count, stable_required)}/{stable_required} frames",
        (1050, 765),
        scale=0.66,
        color=match_color,
    )
    _text(canvas, "184603 expected: 1 connected response region", (1050, 818), scale=0.58)
    _text(canvas, "196FB3 expected: 2 connected response regions", (1050, 855), scale=0.58)

    cv2.line(canvas, (32, 934), (1880, 934), (80, 80, 80), 1)
    _text(
        canvas,
        "TACTILE SOURCE: saved raw UVC -> draft manual warp -> unloaded-baseline absdiff",
        (36, 978),
        scale=0.62,
        color=(210, 210, 210),
        thickness=1,
    )
    _text(
        canvas,
        "Diagnostic only: not SDK WARPED_IMG / DIFF_IMG, not force-calibrated, not a production lift gate.",
        (36, 1022),
        scale=0.57,
        color=(85, 190, 255),
        thickness=1,
    )
    return canvas


def _resolve_fps(manifest: Mapping[str, Any], times: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        return _positive_number(requested, "--fps")
    deltas = np.diff(times)
    measured = 1.0 / float(np.median(deltas))
    stream = manifest.get("stream")
    nominal = stream.get("requested_fps") if isinstance(stream, Mapping) else None
    if isinstance(nominal, (int, float)) and math.isfinite(float(nominal)):
        # Constant-rate output keeps the intended presentation speed while the
        # source timestamps remain in the criteria files.
        return float(nominal)
    return measured


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    capture_dir = args.capture_dir.resolve()
    manifest, frames, times = load_capture(capture_dir)
    specs = load_warp_specs(args.warp_profile)
    operator_outcome = load_operator_outcome(args.operator_outcome)
    events = load_phase_events(args.action_log, float(times[0]))
    phases = phases_for_frames(times, events)
    baseline_rows = baseline_indices(phases, args.baseline_max_frames)

    baseline_a = make_baseline(capture_dir, frames, baseline_rows, "tactile_a", specs[SERIAL_A])
    baseline_b = make_baseline(capture_dir, frames, baseline_rows, "tactile_b", specs[SERIAL_B])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / args.video_name
    if video_path.exists():
        raise ValueError(f"Refusing to overwrite existing video: {video_path}")
    fps = _resolve_fps(manifest, times, args.fps)
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    criteria_rows: list[dict[str, Any]] = []
    stable_count = 0
    shadow_gate_latched = False
    previous_phase: str | None = None
    magnitude_window_a: deque[np.ndarray] = deque(maxlen=args.temporal_window)
    magnitude_window_b: deque[np.ndarray] = deque(maxlen=args.temporal_window)
    try:
        for index, frame in enumerate(frames):
            warped_frame_a = warp_raw(
                _load_bgr(capture_dir / str(frame["tactile_a"])), specs[SERIAL_A]
            )
            warped_frame_b = warp_raw(
                _load_bgr(capture_dir / str(frame["tactile_b"])), specs[SERIAL_B]
            )
            magnitude_window_a.append(_difference_magnitude(warped_frame_a, baseline_a))
            magnitude_window_b.append(_difference_magnitude(warped_frame_b, baseline_b))
            smoothed_a = np.median(np.stack(magnitude_window_a, axis=0), axis=0).astype(
                np.float32
            )
            smoothed_b = np.median(np.stack(magnitude_window_b, axis=0), axis=0).astype(
                np.float32
            )
            metric_a = tactile_metric(
                smoothed_a,
                threshold=args.diff_threshold,
                min_component_area_px=args.min_component_area_px,
            )
            metric_b = tactile_metric(
                smoothed_b,
                threshold=args.diff_threshold,
                min_component_area_px=args.min_component_area_px,
            )
            pair_match = (
                metric_a.component_count == EXPECTED_COMPONENTS_A
                and metric_b.component_count == EXPECTED_COMPONENTS_B
            )
            phase = phases[index]
            if phase != previous_phase:
                stable_count = 0
                previous_phase = phase
            stable_count = stable_count + 1 if pair_match else 0
            if (
                phase in {PHASE_REVIEWED_CLOSE, PHASE_SUCCESS}
                and stable_count >= args.stable_match_frames
            ):
                shadow_gate_latched = True
            rgb = _load_bgr(capture_dir / str(frame["rgb"]))
            canvas = compose_frame(
                rgb,
                warped_frame_a,
                warped_frame_b,
                metric_a,
                metric_b,
                phase=phase,
                elapsed_s=float(times[index] - times[0]),
                pair_match=pair_match,
                stable_count=stable_count,
                stable_required=args.stable_match_frames,
                shadow_gate_latched=shadow_gate_latched,
                operator_outcome=operator_outcome,
            )
            writer.write(canvas)
            badge = badge_for_frame(
                phase,
                pair_match=pair_match,
                shadow_gate_latched=shadow_gate_latched,
                operator_outcome=operator_outcome,
            )
            criteria_rows.append(
                {
                    "frame_index": index,
                    "capture_host_monotonic_s": float(times[index]),
                    "capture_elapsed_s": float(times[index] - times[0]),
                    "phase": phase,
                    "phase_source": "action_log",
                    "serial_184603_expected_components": EXPECTED_COMPONENTS_A,
                    "serial_184603_observed_components": metric_a.component_count,
                    "serial_184603_active_area_px": metric_a.active_area_px,
                    "serial_184603_diff_mean": metric_a.diff_mean,
                    "serial_184603_diff_p95": metric_a.diff_p95,
                    "serial_184603_template_match": metric_a.component_count
                    == EXPECTED_COMPONENTS_A,
                    "serial_196FB3_expected_components": EXPECTED_COMPONENTS_B,
                    "serial_196FB3_observed_components": metric_b.component_count,
                    "serial_196FB3_active_area_px": metric_b.active_area_px,
                    "serial_196FB3_diff_mean": metric_b.diff_mean,
                    "serial_196FB3_diff_p95": metric_b.diff_p95,
                    "serial_196FB3_template_match": metric_b.component_count
                    == EXPECTED_COMPONENTS_B,
                    "observed_pair_template_match": pair_match,
                    "observed_pair_stable_frames": stable_count,
                    "observed_pair_stable": stable_count >= args.stable_match_frames,
                    "shadow_gate_latched": shadow_gate_latched,
                    "video_badge": badge.label,
                    "video_badge_source": badge.source,
                    "operator_outcome": (
                        None if operator_outcome is None else operator_outcome.outcome
                    ),
                    "operator_observed_cup_count": (
                        None
                        if operator_outcome is None
                        else operator_outcome.observed_cup_count
                    ),
                    "diagnostic_scope": "raw_uvc_manual_warp_baseline_absdiff",
                }
            )
    finally:
        writer.release()

    output_stem = Path(args.video_name).stem
    if args.video_name == "cup_tactile_demo.mp4":
        criteria_stem = "tactile_criteria"
        report_name = "render_report.json"
    else:
        criteria_stem = f"{output_stem}_criteria"
        report_name = f"{output_stem}_render_report.json"
    csv_path = output_dir / f"{criteria_stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(criteria_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(criteria_rows)

    criteria_path = output_dir / f"{criteria_stem}.json"
    criteria_document = {
        "schema": SCHEMA,
        "capture_dir": str(capture_dir),
        "action_log": str(args.action_log.resolve()),
        "operator_outcome": (
            None
            if operator_outcome is None
            else {
                "path": str(operator_outcome.path),
                "outcome": operator_outcome.outcome,
                "observed_cup_count": operator_outcome.observed_cup_count,
                "source": operator_outcome.source,
            }
        ),
        "video": str(video_path),
        "criteria_csv": str(csv_path),
        "criteria_json": str(criteria_path),
        "frames": len(criteria_rows),
        "fps": fps,
        "baseline_frame_indices": baseline_rows,
        "diagnostic": {
            "source": "saved raw UVC BGR frames",
            "warp_profile": str(args.warp_profile.resolve()),
            "warp_profile_status": "draft",
            "difference": "Lab Euclidean magnitude from unloaded median baseline",
            "diff_threshold": args.diff_threshold,
            "min_component_area_px": args.min_component_area_px,
            "temporal_window": args.temporal_window,
            "calibrated_classifier": False,
            "sdk_warped_or_diff": False,
            "force_used": False,
        },
        "templates": {
            SERIAL_A: {"label": "1-region template", "expected_components": 1},
            SERIAL_B: {"label": "2-region template", "expected_components": 2},
        },
        "phase_events": [event.__dict__ for event in events],
        "badge_semantics": (
            "An action-log phase or pose/width check cannot create a pass. A pass "
            "requires a stable observed 1/2-region shadow template or an explicit "
            "operator-verified single_cup_pick. Operator-verified failures override "
            "all lift/hold badges. LOW PROBE / RETRY is scripted and does not claim "
            "a sensor rejection."
        ),
        "rows": criteria_rows,
    }
    _write_json(criteria_path, criteria_document)
    _write_json(
        output_dir / report_name,
        {
            "schema": SCHEMA,
            "status": "complete",
            "video": str(video_path),
            "criteria_csv": str(csv_path),
            "criteria_json": str(criteria_path),
            "operator_outcome": (
                None if operator_outcome is None else operator_outcome.outcome
            ),
            "frame_count": len(criteria_rows),
            "resolution": [1920, 1080],
            "fps": fps,
        },
    )
    return criteria_document


def main() -> int:
    args = parse_args()
    result = render(args)
    print(f"Rendered {result['frames']} frames to {result['video']}")
    print(f"Criteria: {result['criteria_csv']}")
    print(f"Criteria metadata: {result['criteria_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
