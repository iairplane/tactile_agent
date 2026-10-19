#!/usr/bin/env python3
"""Convert one Imitrob camera stream to demo2gripper RGB-D inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np


FRAME_PATTERN = re.compile(r"C(?P<camera>[12])F(?P<frame>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera", choices=("C1", "C2"), default="C2")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=3,
        help="Keep every Nth 30 FPS frame; 3 produces a 10 FPS sequence.",
    )
    parser.add_argument(
        "--depth-translation-x-px",
        type=float,
        default=0.0,
        help="Translate depth into the RGB image along x; positive moves right.",
    )
    parser.add_argument(
        "--depth-translation-y-px",
        type=float,
        default=0.0,
        help="Translate depth into the RGB image along y; positive moves down.",
    )
    return parser.parse_args()


def discover_depth_frames(directory: Path, camera: str) -> list[tuple[int, Path]]:
    frames: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = FRAME_PATTERN.fullmatch(path.name)
        if match and f"C{match.group('camera')}" == camera:
            frames.append((int(match.group("frame")), path))
    frames.sort()
    if not frames:
        raise ValueError(f"No {camera} depth frames found in {directory}")
    return frames


def load_intrinsics(parameters_path: Path, camera: str) -> np.ndarray:
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    values = np.asarray(parameters[f"K_{camera}"], dtype=np.float32)
    if values.size != 9:
        raise ValueError(f"K_{camera} must contain nine values, got shape {values.shape}")
    return values.reshape(3, 3)


def main() -> int:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step must be positive")

    sequence_dir = args.sequence_dir.resolve()
    depth_frames = discover_depth_frames(sequence_dir / "Depth", args.camera)
    selected = depth_frames[:: args.frame_step]
    intrinsics = load_intrinsics(sequence_dir / "parameters.json", args.camera)

    image_dir = sequence_dir / "Image"
    video_path = sequence_dir / f"Image_{args.camera}.mp4"
    raw_images_available = all(
        (image_dir / f"{args.camera}F{frame_id:04d}.png").is_file()
        for frame_id, _ in selected
    )
    capture = None
    video_by_frame: dict[int, np.ndarray] = {}
    if not raw_images_available:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open RGB video: {video_path}")
        first_frame_id = depth_frames[0][0]
        selected_ids = {frame_id for frame_id, _ in selected}
        video_index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame_id = first_frame_id + video_index
            if frame_id in selected_ids:
                video_by_frame[frame_id] = image
            video_index += 1
        capture.release()
        if set(video_by_frame) != selected_ids:
            missing = sorted(selected_ids - set(video_by_frame))
            raise ValueError(f"RGB video is missing selected raw frames: {missing}")

    rgb_output = args.output_dir / "rgb"
    depth_output = args.output_dir / "depth"
    rgb_output.mkdir(parents=True, exist_ok=True)
    depth_output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    image_shape: tuple[int, int] | None = None
    for output_index, (frame_id, depth_path) in enumerate(selected):
        image_path = image_dir / f"{args.camera}F{frame_id:04d}.png"
        image = (
            cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw_images_available
            else video_by_frame[frame_id]
        )
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if image is None or depth_mm is None or depth_mm.ndim != 2:
            raise ValueError(f"Could not read paired RGB-D frame F{frame_id:04d}")
        if image.shape[:2] != depth_mm.shape:
            raise ValueError(
                f"RGB/depth shape mismatch at F{frame_id:04d}: "
                f"{image.shape[:2]} versus {depth_mm.shape}"
            )
        if image_shape is None:
            image_shape = image.shape[:2]
        elif image.shape[:2] != image_shape:
            raise ValueError(f"Image shape changed at F{frame_id:04d}")

        name = f"{output_index:05d}"
        rgb_path = rgb_output / f"{name}.png"
        if not cv2.imwrite(str(rgb_path), image):
            raise RuntimeError(f"Could not write {rgb_path}")
        depth_m = depth_mm.astype(np.float32) * 0.001
        if args.depth_translation_x_px or args.depth_translation_y_px:
            depth_m = cv2.warpAffine(
                depth_m,
                np.asarray(
                    [
                        [1.0, 0.0, args.depth_translation_x_px],
                        [0.0, 1.0, args.depth_translation_y_px],
                    ],
                    dtype=np.float32,
                ),
                (image.shape[1], image.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        np.save(depth_output / f"{name}.npy", depth_m, allow_pickle=False)
        records.append(
            {
                "output_frame": output_index,
                "raw_frame": frame_id,
                "rgb": f"rgb/{name}.png",
                "depth": f"depth/{name}.npy",
            }
        )

    np.save(args.output_dir / "K.npy", intrinsics, allow_pickle=False)
    fps = 30.0 / args.frame_step
    manifest = {
        "dataset": "Imitrob v2 powerdrill/metascrewing/S1/left",
        "camera_view": args.camera,
        "source_sequence": str(sequence_dir),
        "rgb_source": "raw PNG" if raw_images_available else video_path.name,
        "depth_source_unit": "millimetre uint16 PNG",
        "output_depth_unit": "metre float32 NPY",
        "depth_registration": {
            "method": (
                "image_translation"
                if args.depth_translation_x_px or args.depth_translation_y_px
                else "as_published"
            ),
            "translation_xy_px": [
                args.depth_translation_x_px,
                args.depth_translation_y_px,
            ],
        },
        "source_fps": 30.0,
        "output_fps": fps,
        "frame_step": args.frame_step,
        "image_size_hw": list(image_shape or ()),
        "intrinsics": "K.npy",
        "frames": records,
    }
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(records)} {args.camera} RGB-D frames at {fps:.1f} FPS "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
