#!/usr/bin/env python3
"""Run RTMPose Hand5 on SAM hand boxes and lift its 21 joints with RGB-D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _paths(directory: Path, suffixes: set[str]) -> list[Path]:
    paths = [
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    ]
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"frame filename stems must be integers in {directory}") from error


def _mask_bbox(mask: np.ndarray) -> list[float] | None:
    rows, columns = np.nonzero(mask)
    if len(columns) == 0:
        return None
    return [
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    ]


def _depth_at(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int,
    hand_mask: np.ndarray,
) -> float | None:
    height, width = depth_m.shape
    center_x, center_y = int(round(u)), int(round(v))
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0.0) & (hand_mask[y0:y1, x0:x1] > 0)
    values = patch[valid]
    if values.size == 0:
        return None
    # Hand keypoints often lie on the silhouette, where registered depth also
    # contains the farther table.  Favor the foreground surface in that mix.
    return float(np.quantile(values, 0.25))


def _backproject(u: float, v: float, depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (u - intrinsics[0, 2]) * depth_m / intrinsics[0, 0],
            (v - intrinsics[1, 2]) * depth_m / intrinsics[1, 1],
            depth_m,
        ],
        dtype=np.float32,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--hand-mask-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="RTMPose Hand5 ONNX model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("onnxruntime", "opencv"), default="onnxruntime")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-radius-px", type=int, default=20)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.depth_radius_px < 0:
        raise ValueError("depth-radius-px must be non-negative")
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    rgb_paths = _paths(args.rgb_dir, IMAGE_SUFFIXES)
    if not rgb_paths:
        raise ValueError(f"no RGB images found in {args.rgb_dir}")
    depth_paths = {path.stem: path for path in _paths(args.depth_dir, {".npy"})}
    mask_paths = {path.stem: path for path in _paths(args.hand_mask_dir, {".png"})}

    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=float)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("intrinsics focal lengths must be positive")

    from rtmlib import RTMPose

    estimator = RTMPose(
        str(args.model.resolve()),
        model_input_size=(256, 256),
        to_openpose=False,
        backend=args.backend,
        device=args.device,
    )

    frame_count = len(rgb_paths)
    keypoints_2d = np.zeros((frame_count, 21, 2), dtype=np.float32)
    scores_2d = np.zeros((frame_count, 21), dtype=np.float32)
    joints_camera_m = np.zeros((frame_count, 21, 3), dtype=np.float32)
    confidence = np.zeros((frame_count, 21), dtype=np.float32)
    frame_records: list[dict[str, object]] = []

    for frame_index, rgb_path in enumerate(rgb_paths):
        depth_path = depth_paths.get(rgb_path.stem)
        mask_path = mask_paths.get(rgb_path.stem)
        if depth_path is None or mask_path is None:
            missing = "depth" if depth_path is None else "hand mask"
            raise ValueError(f"missing {missing} for {rgb_path.name}")

        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        hand_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or hand_mask is None:
            raise ValueError(f"failed to read RGB or mask for {rgb_path.name}")
        height, width = image_bgr.shape[:2]
        if hand_mask.shape != (height, width):
            raise ValueError(f"mask shape mismatch for {rgb_path.name}")
        depth_m = np.asarray(np.load(depth_path, allow_pickle=False), dtype=float)
        if depth_m.shape != (height, width):
            raise ValueError(f"depth shape mismatch for {rgb_path.name}")

        bbox = _mask_bbox(hand_mask)
        if bbox is None:
            frame_records.append({"frame": rgb_path.name, "status": "empty_hand_mask"})
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predicted_keypoints, predicted_scores = estimator(image_rgb, bboxes=[bbox])
        predicted_keypoints = np.asarray(predicted_keypoints, dtype=np.float32)
        predicted_scores = np.asarray(predicted_scores, dtype=np.float32)
        if predicted_keypoints.shape != (1, 21, 2) or predicted_scores.shape != (1, 21):
            raise RuntimeError(
                "RTMPose Hand5 must return keypoints (1, 21, 2) and scores (1, 21)"
            )

        points = predicted_keypoints[0]
        scores = predicted_scores[0]
        keypoints_2d[frame_index] = points
        scores_2d[frame_index] = scores
        valid_depth = 0
        for joint_index, (point, score) in enumerate(zip(points, scores, strict=True)):
            depth = _depth_at(
                depth_m,
                float(point[0]),
                float(point[1]),
                args.depth_radius_px,
                hand_mask,
            )
            if depth is None:
                continue
            joints_camera_m[frame_index, joint_index] = _backproject(
                float(point[0]), float(point[1]), depth, intrinsics
            )
            confidence[frame_index, joint_index] = float(np.clip(score, 0.0, 1.0))
            valid_depth += 1

        frame_records.append(
            {
                "frame": rgb_path.name,
                "status": "ok" if valid_depth else "no_valid_joint_depth",
                "bbox_xyxy": bbox,
                "valid_depth_joints": valid_depth,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "hand_keypoints_2d.npy", keypoints_2d)
    np.save(args.output_dir / "hand_keypoint_scores_2d.npy", scores_2d)
    np.save(args.output_dir / "hand_joints_camera_m.npy", joints_camera_m)
    np.save(args.output_dir / "hand_joint_confidence.npy", confidence)
    np.save(
        args.output_dir / "timestamps_s.npy",
        np.arange(frame_count, dtype=np.float64) / args.fps,
    )
    (args.output_dir / "rtmpose_manifest.json").write_text(
        json.dumps(
            {
                "backend": "rtmpose-hand5-rgbd",
                "display_name": "RTMPose-m Hand5 + RGB-D",
                "model": str(args.model.resolve()),
                "keypoint_order": "wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4",
                "score_contract": {
                    "hand_keypoint_scores_2d.npy": "raw RTMPose SimCC maxima",
                    "hand_joint_confidence.npy": "raw score clipped to [0, 1], or 0 without valid depth",
                },
                "frames": frame_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
