#!/usr/bin/env python3
"""Compile a review-only top-cup rim/pinch relation from one RGB-D demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from demo2gripper.cup_rim_relation import (
    CupRimRelationArtifacts,
    CupRimRelationConfig,
    compile_cup_rim_relation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="(F,H,W) .npy full top-cup masks or directory of numeric mask PNGs",
    )
    parser.add_argument("--hand-pose-dir", type=Path, required=True)
    parser.add_argument("--stages-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rim-ellipses",
        type=Path,
        default=None,
        help=(
            "optional (F,5) cx,cy,major_radius,minor_radius,major_angle_rad .npy; "
            "otherwise fit each selected mask"
        ),
    )
    parser.add_argument("--max-precontact-frames", type=int, default=8)
    return parser.parse_args()


def _numeric_paths(directory: Path, suffixes: set[str]) -> list[Path]:
    paths = [path for path in directory.iterdir() if path.suffix.lower() in suffixes]
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"all files in {directory} must have numeric stems") from error


def _load_stages(path: Path, frame_count: int) -> dict[str, tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages: dict[str, tuple[int, int]] = {}
    if payload.get("schema") == "demo2gripper.image_only_semantic_stages.v1":
        for item in payload.get("stages", []):
            # This image-review schema records inclusive end_frame values.
            stages[str(item["name"])] = (
                int(item["start_frame"]),
                int(item["end_frame"]) + 1,
            )
    elif isinstance(payload.get("stage_intervals"), dict):
        stages = {
            str(name): tuple(int(value) for value in interval)
            for name, interval in payload["stage_intervals"].items()
        }
    else:
        raise ValueError(
            "stages JSON must be image_only_semantic_stages.v1 or contain stage_intervals"
        )
    for name in ("PRESHAPE", "GRASP"):
        if name not in stages:
            raise ValueError(f"stages JSON is missing {name}")
        start, end = stages[name]
        if not 0 <= start < end <= frame_count:
            raise ValueError(f"{name} lies outside the capture")
    return stages


class MaskSource:
    def __init__(self, path: Path, frame_count: int):
        self.archive: np.ndarray | None = None
        self.paths: list[Path] | None = None
        if path.is_file():
            self.archive = np.load(path, mmap_mode="r", allow_pickle=False)
            if self.archive.ndim != 3 or len(self.archive) != frame_count:
                raise ValueError("mask array must have shape (F, H, W)")
        elif path.is_dir():
            self.paths = _numeric_paths(path, {".png", ".jpg", ".jpeg"})
            if len(self.paths) != frame_count:
                raise ValueError("mask image count must match RGB frame count")
        else:
            raise FileNotFoundError(path)

    def frame(self, index: int) -> np.ndarray:
        if self.archive is not None:
            return np.asarray(self.archive[index], dtype=bool)
        assert self.paths is not None
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read mask {self.paths[index]}")
        return image > 0


def _fit_mask_ellipse(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("top-cup mask has no contour")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        raise ValueError("top-cup mask contour has fewer than five points")
    (cx, cy), (diameter_a, diameter_b), angle_deg = cv2.fitEllipse(contour)
    if diameter_a >= diameter_b:
        major_radius = 0.5 * diameter_a
        minor_radius = 0.5 * diameter_b
        major_angle_deg = angle_deg
    else:
        major_radius = 0.5 * diameter_b
        minor_radius = 0.5 * diameter_a
        major_angle_deg = angle_deg + 90.0
    return np.array(
        [
            cx,
            cy,
            major_radius,
            minor_radius,
            np.deg2rad(major_angle_deg),
        ],
        dtype=float,
    )


def _project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points[:, 2]
    pixels = np.full((len(points), 2), np.nan)
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    pixels[valid, 0] = K[0, 0] * points[valid, 0] / z[valid] + K[0, 2]
    pixels[valid, 1] = K[1, 1] * points[valid, 1] / z[valid] + K[1, 2]
    return pixels


def _point(pixel: np.ndarray) -> tuple[int, int]:
    return int(round(float(pixel[0]))), int(round(float(pixel[1])))


def _write_preview(
    path: Path,
    artifacts: CupRimRelationArtifacts,
    rgb_paths: list[Path],
    masks: MaskSource,
    hand_keypoints_2d: np.ndarray,
    K: np.ndarray,
) -> None:
    pre_index = artifacts.representative_precontact_frame
    grasp_index = artifacts.representative_grasp_frame
    panels: list[np.ndarray] = []
    for index in (pre_index, grasp_index):
        image = cv2.imread(str(rgb_paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read RGB frame {rgb_paths[index]}")
        panels.append(image)
    pre, grasp = panels
    contour_mask = (masks.frame(pre_index).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pre, contours, -1, (0, 220, 255), 1)

    T = artifacts.T_camera_rim
    theta = np.linspace(0.0, 2.0 * np.pi, 100)
    rim_local = np.column_stack(
        (
            artifacts.rim_radius_m * np.cos(theta),
            artifacts.rim_radius_m * np.sin(theta),
            np.zeros_like(theta),
        )
    )
    rim_camera = rim_local @ T[:3, :3].T + T[:3, 3]
    rim_pixels = _project(rim_camera, K)
    finite = np.isfinite(rim_pixels).all(axis=1)
    if np.count_nonzero(finite) >= 2:
        cv2.polylines(
            pre,
            [np.rint(rim_pixels[finite]).astype(np.int32)],
            True,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    center_and_axis = np.stack((T[:3, 3], T[:3, 3] + 0.8 * artifacts.rim_radius_m * T[:3, 2]))
    center_axis_px = _project(center_and_axis, K)
    if np.isfinite(center_axis_px).all():
        cv2.arrowedLine(
            pre,
            _point(center_axis_px[0]),
            _point(center_axis_px[1]),
            (255, 80, 0),
            3,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        cv2.circle(pre, _point(center_axis_px[0]), 5, (0, 0, 255), -1)

    thumb = hand_keypoints_2d[grasp_index, 4]
    index = hand_keypoints_2d[grasp_index, 8]
    if np.isfinite(np.concatenate((thumb, index))).all():
        cv2.line(grasp, _point(thumb), _point(index), (255, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(grasp, _point(thumb), 6, (0, 255, 255), -1)
        cv2.circle(grasp, _point(index), 6, (0, 255, 0), -1)
    labels = (
        (pre, f"PRECONTACT f{pre_index}: frozen red rim / blue cup axis"),
        (grasp, f"GRASP f{grasp_index}: thumb-index closing line"),
    )
    for image, label in labels:
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    canvas = np.concatenate((pre, grasp), axis=1)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"could not write {path}")


def main() -> None:
    args = parse_args()
    if args.max_precontact_frames < 1:
        raise ValueError("--max-precontact-frames must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty {args.output_dir}")
    rgb_paths = _numeric_paths(args.input_dir / "rgb", {".png", ".jpg", ".jpeg"})
    depth_paths = _numeric_paths(args.input_dir / "depth", {".npy"})
    if not rgb_paths or len(rgb_paths) != len(depth_paths):
        raise ValueError("RGB and metric-depth frame counts must match and be non-zero")
    frame_count = len(rgb_paths)
    stages = _load_stages(args.stages_json, frame_count)
    # The beginning of APPROACH is already pre-contact and is typically much
    # less occluded than PRESHAPE.  Fall back to PRESHAPE when APPROACH is not
    # available in an externally supplied stage contract.
    precontact_interval = stages.get("APPROACH", stages["PRESHAPE"])
    selected = np.arange(
        precontact_interval[0],
        min(precontact_interval[1], precontact_interval[0] + args.max_precontact_frames),
    )
    mask_source = MaskSource(args.masks, frame_count)
    depths = np.stack(
        [np.load(depth_paths[index], allow_pickle=False) for index in selected]
    )
    masks = np.stack([mask_source.frame(int(index)) for index in selected])
    K = np.asarray(np.load(args.input_dir / "K.npy", allow_pickle=False), dtype=float)
    joints = np.asarray(
        np.load(args.hand_pose_dir / "hand_joints_camera_m.npy", allow_pickle=False),
        dtype=float,
    )
    confidence = np.asarray(
        np.load(args.hand_pose_dir / "hand_joint_confidence.npy", allow_pickle=False),
        dtype=float,
    )
    keypoints = np.asarray(
        np.load(args.hand_pose_dir / "hand_keypoints_2d.npy", allow_pickle=False),
        dtype=float,
    )
    if len(joints) != frame_count or keypoints.shape != (frame_count, 21, 2):
        raise ValueError("hand pose arrays must match the RGB capture")
    ellipses: np.ndarray
    ellipse_source: str
    if args.rim_ellipses is not None:
        all_ellipses = np.load(args.rim_ellipses, mmap_mode="r", allow_pickle=False)
        if all_ellipses.shape != (frame_count, 5):
            raise ValueError("full rim ellipse array must have shape (F, 5)")
        ellipses = np.asarray(all_ellipses[selected], dtype=float)
        ellipse_source = str(args.rim_ellipses.resolve())
    else:
        ellipses = np.stack([_fit_mask_ellipse(mask) for mask in masks])
        ellipse_source = "fitEllipse(full_top_cup_mask)"

    artifacts = compile_cup_rim_relation(
        precontact_depth_m=depths,
        precontact_masks=masks,
        precontact_frame_indices=selected,
        K=K,
        hand_joints_camera_m=joints,
        hand_joint_confidence=confidence,
        stage_intervals=stages,
        rim_ellipses_px=ellipses,
        config=CupRimRelationConfig(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = args.output_dir / "cup_rim_geometry.npz"
    np.savez_compressed(
        geometry_path,
        T_camera_rim=artifacts.T_camera_rim,
        T_rim_human_pinch_tcp=artifacts.T_rim_human_pinch_tcp,
        T_camera_human_pinch_tcp=artifacts.T_camera_human_pinch_tcp,
        rim_points_camera_m=artifacts.rim_points_camera_m,
        precontact_frame_indices=artifacts.precontact_frame_indices,
        valid_precontact_frame_indices=artifacts.valid_precontact_frame_indices,
        grasp_frame_indices=artifacts.grasp_frame_indices,
    )
    preview_path = args.output_dir / "preview.png"
    _write_preview(
        preview_path, artifacts, rgb_paths, mask_source, keypoints, K
    )
    payload = {
        "schema": "demo2gripper.cup_rim_relation.v1",
        "status": "review_only",
        "execution_authorized": False,
        "relation_role": "human_demo_geometry_only",
        "frame_convention": (
            "T_parent_child; metres; rim +X local radial toward demonstrated pinch, "
            "+Y tangent, +Z cup opening axis; TCP +X closing and +Z wrist-to-pinch"
        ),
        "T_camera_rim": artifacts.T_camera_rim.tolist(),
        "T_rim_human_pinch_tcp": artifacts.T_rim_human_pinch_tcp.tolist(),
        "rim_radius_m": artifacts.rim_radius_m,
        "demonstrated_pinch_width_m": artifacts.pinch_width_m,
        "stage_intervals_half_open": {name: list(value) for name, value in stages.items()},
        "quality": artifacts.quality,
        "artifacts": {
            "geometry": geometry_path.name,
            "preview": preview_path.name,
        },
        "source": {
            "input_dir": str(args.input_dir.resolve()),
            "masks": str(args.masks.resolve()),
            "hand_pose_dir": str(args.hand_pose_dir.resolve()),
            "stages_json": str(args.stages_json.resolve()),
            "rim_ellipses": (
                ellipse_source
            ),
        },
        "required_before_execution": [
            "operator review of preview.png and quality metrics",
            "fresh live top-cup rim grounding",
            "Hand-E embodiment and TCP adaptation",
            "IK, collision, trajectory, workspace, and tactile gates",
        ],
        "limitations": [
            "The precontact top rim is frozen; grasp-time mask centroids are not used.",
            "The cup opening axis sign assumes that the opening faces the camera.",
            "This proposal does not contain robot targets, joint commands, or execution logic.",
        ],
    }
    relation_path = args.output_dir / "cup_rim_relation.json"
    relation_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {relation_path}")
    print(f"wrote {geometry_path}")
    print(f"wrote {preview_path}")
    print("status: review_only; execution_authorized=false")


if __name__ == "__main__":
    main()
