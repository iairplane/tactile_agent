#!/usr/bin/env python3
"""Run WiLoR and anchor its MANO joints to calibrated RGB-D measurements."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PALM_JOINTS = (0, 5, 9, 13, 17)


def _paths(directory: Path, suffixes: set[str]) -> list[Path]:
    paths = [
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    ]
    try:
        return sorted(paths, key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError(f"frame filename stems must be integers in {directory}") from error


def _depth_at(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int,
    hand_mask: np.ndarray | None,
) -> float | None:
    height, width = depth_m.shape
    center_x, center_y = int(round(u)), int(round(v))
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    patch = depth_m[y0:y1, x0:x1]
    valid = np.isfinite(patch) & (patch > 0.0)
    if hand_mask is not None:
        valid &= hand_mask[y0:y1, x0:x1] > 0
    values = patch[valid]
    if values.size == 0:
        return None
    return float(np.median(values))


def _backproject(u: float, v: float, depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (u - intrinsics[0, 2]) * depth_m / intrinsics[0, 0],
            (v - intrinsics[1, 2]) * depth_m / intrinsics[1, 1],
            depth_m,
        ]
    )


def _project_model(
    joints: np.ndarray, camera_translation: np.ndarray, focal_length: float, width: int, height: int
) -> np.ndarray:
    camera_joints = joints + camera_translation
    pixels = np.empty((len(joints), 2), dtype=float)
    pixels[:, 0] = focal_length * camera_joints[:, 0] / camera_joints[:, 2] + width / 2.0
    pixels[:, 1] = focal_length * camera_joints[:, 1] / camera_joints[:, 2] + height / 2.0
    return pixels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True, help="One metre-valued .npy per RGB frame")
    parser.add_argument("--intrinsics", type=Path, required=True, help="3x3 OpenCV K matrix in .npy")
    parser.add_argument("--hand-mask-dir", type=Path, help="Optional uint8 hand masks with RGB stems")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wilor-root", type=Path, default=Path("third_party/WiLoR"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--detector-confidence", type=float, default=0.3)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--depth-radius-px", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    wilor_root = args.wilor_root.resolve()
    checkpoint = args.checkpoint.resolve()
    model_config = args.model_config.resolve()
    detector_checkpoint = args.detector_checkpoint.resolve()
    sys.path.insert(0, str(wilor_root))
    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full

    rgb_paths = _paths(args.rgb_dir, IMAGE_SUFFIXES)
    if not rgb_paths:
        raise ValueError(f"no RGB images found in {args.rgb_dir}")
    depth_paths = {path.stem: path for path in _paths(args.depth_dir, {".npy"})}
    missing_depth = [path.name for path in rgb_paths if path.stem not in depth_paths]
    if missing_depth:
        raise ValueError(f"missing depth arrays for: {', '.join(missing_depth[:5])}")

    intrinsics = np.asarray(np.load(args.intrinsics, allow_pickle=False), dtype=float)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
    mask_paths = (
        {path.stem: path for path in _paths(args.hand_mask_dir, {".png"})}
        if args.hand_mask_dir
        else {}
    )

    device = torch.device(args.device)
    original_directory = Path.cwd()
    # load_wilor resolves MANO paths relative to the directory containing
    # pretrained_models (the AnyHand or standalone WiLoR repository root).
    os.chdir(model_config.parent.parent)
    try:
        model, model_cfg = load_wilor(
            checkpoint_path=str(checkpoint), cfg_path=str(model_config)
        )
    finally:
        os.chdir(original_directory)
    detector = YOLO(str(detector_checkpoint))
    model = model.to(device).eval()
    detector = detector.to(device)

    frame_count = len(rgb_paths)
    joints_camera_m = np.zeros((frame_count, 21, 3), dtype=np.float32)
    confidence = np.zeros((frame_count, 21), dtype=np.float32)
    requested_side = 1 if args.hand_side == "right" else 0
    frame_records: list[dict[str, object]] = []

    for frame_index, rgb_path in enumerate(rgb_paths):
        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"failed to read RGB image: {rgb_path}")
        height, width = image_bgr.shape[:2]
        depth_m = np.asarray(np.load(depth_paths[rgb_path.stem], allow_pickle=False), dtype=float)
        if depth_m.shape != (height, width):
            raise ValueError(
                f"{depth_paths[rgb_path.stem]} has shape {depth_m.shape}, expected {(height, width)}"
            )
        hand_mask = None
        if args.hand_mask_dir:
            mask_path = mask_paths.get(rgb_path.stem)
            if mask_path is None:
                raise ValueError(f"missing hand mask for {rgb_path.name}")
            hand_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if hand_mask is None or hand_mask.shape != (height, width):
                raise ValueError(f"invalid hand mask: {mask_path}")

        result = detector(image_bgr, conf=args.detector_confidence, verbose=False)[0]
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        scores = result.boxes.conf.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        matching = np.flatnonzero(classes == requested_side)
        if matching.size == 0:
            frame_records.append({"frame": rgb_path.name, "status": "no_matching_hand"})
            continue
        detection_index = int(matching[np.argmax(scores[matching])])
        box = boxes[detection_index : detection_index + 1]
        side = np.array([requested_side], dtype=np.float32)
        detection_score = float(scores[detection_index])

        dataset = ViTDetDataset(
            model_cfg,
            image_bgr,
            box,
            side,
            rescale_factor=args.rescale_factor,
            fp16=False,
        )
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
        batch = recursive_to(batch, device)
        with torch.inference_mode():
            output = model(batch)

        multiplier = 2 * batch["right"] - 1
        predicted_camera = output["pred_cam"].clone()
        predicted_camera[:, 1] = multiplier * predicted_camera[:, 1]
        image_size = batch["img_size"].float()
        scaled_focal = (
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * image_size.max()
        )
        camera_translation = cam_crop_to_full(
            predicted_camera,
            batch["box_center"].float(),
            batch["box_size"].float(),
            image_size,
            scaled_focal,
        )[0].detach().cpu().numpy()
        joints = output["pred_keypoints_3d"][0].detach().cpu().numpy()
        joints[:, 0] *= 2 * requested_side - 1
        pixels = _project_model(
            joints,
            camera_translation,
            float(scaled_focal.detach().cpu().item()),
            width,
            height,
        )

        offsets = []
        for joint_index in PALM_JOINTS:
            depth = _depth_at(
                depth_m,
                pixels[joint_index, 0],
                pixels[joint_index, 1],
                args.depth_radius_px,
                hand_mask,
            )
            if depth is not None:
                offsets.append(
                    _backproject(
                        pixels[joint_index, 0], pixels[joint_index, 1], depth, intrinsics
                    )
                    - joints[joint_index]
                )
        if len(offsets) < 2:
            frame_records.append(
                {"frame": rgb_path.name, "status": "insufficient_depth_anchors"}
            )
            continue

        metric_translation = np.median(np.asarray(offsets), axis=0)
        joints_camera_m[frame_index] = joints + metric_translation
        supported = np.array(
            [
                _depth_at(
                    depth_m, pixel[0], pixel[1], args.depth_radius_px, hand_mask
                )
                is not None
                for pixel in pixels
            ]
        )
        confidence[frame_index] = np.where(
            supported, detection_score, 0.25 * detection_score
        )
        frame_records.append(
            {
                "frame": rgb_path.name,
                "status": "ok",
                "depth_anchors": len(offsets),
                "detector_confidence": detection_score,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "hand_joints_camera_m.npy", joints_camera_m)
    np.save(args.output_dir / "hand_joint_confidence.npy", confidence)
    np.save(
        args.output_dir / "timestamps_s.npy",
        np.arange(frame_count, dtype=np.float64) / args.fps,
    )
    (args.output_dir / "wilor_manifest.json").write_text(
        json.dumps(
            {
                "hand_side": args.hand_side,
                "rgb_frames": [path.name for path in rgb_paths],
                "frames": frame_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
