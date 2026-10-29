#!/usr/bin/env python3
"""Read-only held-contact monitor: saved unloaded WARPED reference, no re-zero.

No robot/gripper command or SDK calibration is issued.  This observes contact
image persistence; it does not estimate force, slip, tube pose or extraction.
Consumers require status=running, bilateral_contact and fresh observed_at_monotonic_s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demo2gripper.test_tube_contact_approach import _largest_component_area  # noqa: E402
from demo2gripper.test_tube_tactile_supervisor import compensated_residual  # noqa: E402


def classify_warped_contact(image, reference, roi, threshold, min_area):
    image, reference, roi = np.asarray(image), np.asarray(reference), np.asarray(roi, dtype=bool)
    if image.shape != reference.shape or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("WARPED image/reference shape mismatch")
    if image.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError("WARPED images must be uint8")
    if roi.shape != image.shape[:2] or not roi.any():
        raise ValueError("invalid contact ROI")
    if not math.isfinite(threshold) or threshold <= 0 or min_area < 1:
        raise ValueError("invalid saved contact thresholds")
    residual = compensated_residual(image, reference, roi)
    area = _largest_component_area(residual >= threshold, roi)
    return {"contact": bool(area >= min_area), "component_area_px": int(area),
            "residual_p99": float(np.quantile(residual[roi], .99))}


def _atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    from diagnose_gf220_sdk import (
        DEFAULT_PROFILE,
        _collect_new,
        _discover_devices,
        _read_profile,
        _write_png,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-run", "--baseline-dir", dest="baseline_dir",
                        type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--duration-s", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        raise ValueError("duration must be positive")
    profile = _read_profile(args.profile)
    metadata = json.loads((args.baseline_dir / "unloaded_contact_profile.json").read_text())
    if metadata.get("schema") != "demo2gripper.test_tube_contact_approach.v1":
        raise ValueError("unrecognized saved contact profile")
    fingers = metadata["fingers"]
    sensors = profile["sensors"]
    serials = [item["sdk_serial"] for item in sensors]
    if len(fingers) != 2 or set(serials) != {item["serial"] for item in fingers}:
        raise ValueError("saved/live sensor serials differ")
    references = {}
    with np.load(args.baseline_dir / "unloaded_contact_profile.npz", allow_pickle=False) as saved:
        for index, finger in enumerate(fingers):
            references[finger["serial"]] = (
                saved[f"finger_{index}_warped_reference"].copy(),
                saved[f"finger_{index}_roi_mask"].copy(),
                float(finger["warped_residual_threshold"]), int(finger["min_component_area_px"]),
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    latest = {
        "schema": "demo2gripper.held_warped_contact_monitor.v1", "status": "starting",
        "bilateral_contact": False, "fresh_bilateral_pairs": 0, "sequence": 0,
        "reference_session_id": metadata["session_id"],
        "reference_directory": str(args.baseline_dir.resolve()),
        "profile_path": str(args.profile.resolve()),
        "products": ["WARPED_IMG"], "sdk_calibration_restored": False,
        "force_or_slip_estimated": False, "observed_at_monotonic_s": time.monotonic(),
    }
    path = args.output_dir / "latest.json"
    _atomic_json(path, latest)
    devices, final_images = [], {}
    try:
        from pyvitaisdk import GF225, GF225OutputProfile, GFDataType, VTSDeviceFinder
        from pyvitaisdk.vts.core.base.profile import VideoStreamFmt

        finder, handles = _discover_devices(serials, 10, .5, VTSDeviceFinder)
        stream = profile["stream"]
        previous = {serial: (None, None) for serial in serials}
        for spec in sensors:
            serial = spec["sdk_serial"]
            device = GF225(config=handles[serial], stream_format=VideoStreamFmt(**stream),
                           output_format=GF225OutputProfile.W240_H240)
            devices.append((spec, device))
        raw_types = (GFDataType.TIME_STAMP, GFDataType.RAW_IMG)
        for _ in range(8):
            for spec, device in devices:
                serial = spec["sdk_serial"]
                _, stamp, crc = _collect_new(device, raw_types, *raw_types,
                                             *previous[serial], 2.0)
                previous[serial] = (stamp, crc)
        for spec, device in devices:
            device._image_processor.set_warp_params(corner_points=spec["corners_xy"], mode="manual")
        types = (*raw_types, GFDataType.WARPED_IMG)
        deadline = time.monotonic() + args.duration_s
        streak = 0
        while time.monotonic() < deadline:
            pads, capture_times = [], []
            for spec, device in devices:
                serial = spec["sdk_serial"]
                sample, stamp, crc = _collect_new(device, types, *raw_types,
                                                 *previous[serial], 2.0)
                captured = time.monotonic()
                if previous[serial][0] is not None and stamp <= previous[serial][0]:
                    raise RuntimeError(f"{serial}: SDK timestamp went backwards")
                previous[serial] = (stamp, crc)
                warped = np.asarray(sample[GFDataType.WARPED_IMG]).copy()
                if warped.shape != (240, 240, 3) or warped.std() < 5:
                    raise RuntimeError(f"{serial}: invalid manual WARPED image")
                evidence = classify_warped_contact(warped, *references[serial])
                pads.append({"serial": serial, "sdk_timestamp_raw": stamp,
                             "capture_monotonic_s": captured, **evidence})
                capture_times.append(captured)
                final_images[spec["name"]] = warped
                if latest["sequence"] == 0:
                    _write_png(args.output_dir / f"{spec['name']}_before_warped.png", warped)
            skew = max(capture_times) - min(capture_times)
            valid = skew <= .08
            streak = streak + 1 if valid and all(pad["contact"] for pad in pads) else 0
            latest.update(status="running", observed_at_monotonic_s=time.monotonic(), pads=pads,
                          sequence=latest["sequence"] + 1, pair_skew_s=skew,
                          valid=valid, fresh_bilateral_pairs=streak, bilateral_contact=streak >= 2)
            _atomic_json(path, latest)
        latest.update(status="completed", bilateral_contact=False,
                      observed_at_monotonic_s=time.monotonic())
        _atomic_json(path, latest)
        return 0
    except BaseException as exc:
        latest.update(status="error", bilateral_contact=False,
                      observed_at_monotonic_s=time.monotonic(),
                      error=f"{type(exc).__name__}: {exc}")
        _atomic_json(path, latest)
        raise
    finally:
        for name, image in final_images.items():
            _write_png(args.output_dir / f"{name}_after_warped.png", image)
        for _, device in devices:
            device.release()


if __name__ == "__main__":
    raise SystemExit(main())
