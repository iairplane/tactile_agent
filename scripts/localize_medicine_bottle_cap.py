#!/usr/bin/env python3
"""Review-only RGB-D localization for the current upright black-cap bottle scene.

This command only reads one aligned RGB-D frame and writes files.  It has no ROS,
Franka, Hand-E, or tactile command path.  The explicit ROI is part of the current
scene definition and must tightly contain the cap without clipping it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from demo2gripper.medicine_bottle import parse_bottle_cap_frame_payload
from demo2gripper.medicine_bottle_localizer import (
    CurrentBottleCapConfig,
    draw_bottle_cap_overlay,
    localize_current_bottle_cap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path, help="Aligned depth .npy")
    parser.add_argument("--intrinsics", required=True, type=Path, help="OpenCV K.npy")
    parser.add_argument("--t-base-camera", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--roi",
        required=True,
        type=int,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX_EXCLUSIVE", "Y_MAX_EXCLUSIVE"),
    )
    parser.add_argument(
        "--depth-scale-m-per-unit",
        type=float,
        help="Required for integer/Z16 depth; omit for metre-valued floating depth",
    )
    parser.add_argument("--max-dark-value", type=int, default=95)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--min-component-pixels", type=int, default=120)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.50)
    parser.add_argument("--min-dominance-ratio", type=float, default=1.35)
    parser.add_argument("--morphology-radius-px", type=int, default=2)
    parser.add_argument("--component-margin-px", type=int, default=1)
    parser.add_argument("--min-cap-diameter-m", type=float, default=0.012)
    parser.add_argument("--max-cap-diameter-m", type=float, default=0.080)
    parser.add_argument("--robust-quantile", type=float, default=0.05)
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--cap-frame", default="medicine_cap_axis")
    parser.add_argument("--tcp-frame", default="hande_tcp")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_depth(path: Path, scale: float | None) -> np.ndarray:
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"depth must be two-dimensional, got {depth.shape}")
    if np.issubdtype(depth.dtype, np.integer):
        if scale is None:
            raise ValueError("integer/Z16 depth requires --depth-scale-m-per-unit")
    elif not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"unsupported depth dtype: {depth.dtype}")
    if scale is not None:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("depth scale must be finite and positive")
        depth = depth.astype(np.float64) * scale
    return depth.astype(np.float64, copy=False)


def _write_outputs(
    output_dir: Path,
    *,
    cap_frame: dict[str, object],
    report: dict[str, object],
    mask: np.ndarray,
    overlay: np.ndarray,
    overwrite: bool,
) -> None:
    output = output_dir.resolve()
    if output.exists() and not overwrite:
        raise ValueError(f"output directory already exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        with (staging / "medicine_bottle_cap_frame.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(cap_frame, handle, indent=2, allow_nan=False)
            handle.write("\n")
        with (staging / "localization_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if not cv2.imwrite(str(staging / "cap_mask.png"), mask.astype(np.uint8) * 255):
            raise RuntimeError("failed to write cap_mask.png")
        if not cv2.imwrite(str(staging / "diagnostic_overlay.png"), overlay):
            raise RuntimeError("failed to write diagnostic_overlay.png")
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> Path:
    rgb_path = args.rgb.resolve()
    depth_path = args.depth.resolve()
    intrinsics_path = args.intrinsics.resolve()
    transform_path = args.t_base_camera.resolve()
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read RGB image: {rgb_path}")
    depth = _load_depth(depth_path, args.depth_scale_m_per_unit)
    intrinsics = np.load(intrinsics_path, allow_pickle=False)
    T_base_camera = np.load(transform_path, allow_pickle=False)
    config = CurrentBottleCapConfig(
        roi_xyxy_px=tuple(args.roi),
        max_dark_value=args.max_dark_value,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        min_component_pixels=args.min_component_pixels,
        min_valid_depth_ratio=args.min_valid_depth_ratio,
        min_dominance_ratio=args.min_dominance_ratio,
        morphology_radius_px=args.morphology_radius_px,
        component_margin_px=args.component_margin_px,
        min_cap_diameter_m=args.min_cap_diameter_m,
        max_cap_diameter_m=args.max_cap_diameter_m,
        robust_quantile=args.robust_quantile,
    )
    result = localize_current_bottle_cap(
        image,
        depth,
        intrinsics,
        T_base_camera,
        config,
        base_frame=args.base_frame,
        cap_frame=args.cap_frame,
        tcp_frame=args.tcp_frame,
    )
    cap_frame = result.frame.as_dict()
    # Re-parse the serialized payload so the written artifact is proven to satisfy
    # the strict contract rather than only an internal dataclass construction.
    parse_bottle_cap_frame_payload(cap_frame, expected_base_frame=args.base_frame)
    report = result.report_dict()
    report["inputs"] = {
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "intrinsics": str(intrinsics_path),
        "T_base_camera": str(transform_path),
        "depth_scale_m_per_unit": args.depth_scale_m_per_unit,
    }
    report["configuration"] = {
        "max_dark_value": config.max_dark_value,
        "depth_range_m": [config.min_depth_m, config.max_depth_m],
        "min_component_pixels": config.min_component_pixels,
        "min_valid_depth_ratio": config.min_valid_depth_ratio,
        "min_dominance_ratio": config.min_dominance_ratio,
        "morphology_radius_px": config.morphology_radius_px,
        "component_margin_px": config.component_margin_px,
        "cap_diameter_range_m": [
            config.min_cap_diameter_m,
            config.max_cap_diameter_m,
        ],
        "robust_quantile": config.robust_quantile,
    }
    overlay = draw_bottle_cap_overlay(image, result)
    output_dir = args.output_dir.resolve()
    _write_outputs(
        output_dir,
        cap_frame=cap_frame,
        report=report,
        mask=result.selected_mask,
        overlay=overlay,
        overwrite=args.overwrite,
    )
    return output_dir


def main() -> int:
    try:
        output = run(parse_args())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
