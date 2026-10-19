#!/usr/bin/env python3
"""Propagate first-frame multi-object prompts through an RGB sequence with SAM 2."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image


RGB_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        required=True,
        help="Hydra config name, e.g. configs/sam2.1/sam2.1_hiera_l.yaml.",
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--async-loading-frames",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def discover_frames(rgb_dir: Path) -> tuple[list[Path], tuple[int, int]]:
    if not rgb_dir.is_dir():
        raise ValueError(f"RGB directory does not exist: {rgb_dir}")

    frames = sorted(
        (path for path in rgb_dir.iterdir() if path.suffix.lower() in RGB_SUFFIXES),
        key=lambda path: _numeric_stem(path),
    )
    if not frames:
        raise ValueError(f"No PNG/JPEG frames found in {rgb_dir}")

    frame_numbers = [_numeric_stem(path) for path in frames]
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("RGB frame numbers are not unique")
    expected = list(range(frame_numbers[0], frame_numbers[-1] + 1))
    if frame_numbers != expected:
        raise ValueError(
            f"RGB frame numbers must be contiguous; got {frame_numbers[0]}.."
            f"{frame_numbers[-1]} with {len(frame_numbers)} files"
        )

    image_size: tuple[int, int] | None = None
    for path in frames:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"RGB frame must have PIL mode RGB: {path} has {image.mode}")
            current_size = (image.height, image.width)
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                f"All RGB frames must have one size; {path} is {current_size}, "
                f"expected {image_size}"
            )

    assert image_size is not None
    return frames, image_size


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Frame filename stem must be an integer: {path.name}") from exc


def load_prompts(path: Path, height: int, width: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) - {"frame_idx", "objects"}:
        raise ValueError("Prompts JSON must contain only frame_idx and objects")
    if payload.get("frame_idx", 0) != 0:
        raise ValueError("This adapter accepts first-frame prompts only (frame_idx must be 0)")
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Prompts JSON objects must be a non-empty list")

    parsed: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{index}] must be an object")
        unknown = set(item) - {"obj_id", "points", "labels", "box"}
        if unknown:
            raise ValueError(f"objects[{index}] has unsupported keys: {sorted(unknown)}")

        obj_id = item.get("obj_id")
        if isinstance(obj_id, bool) or not isinstance(obj_id, int) or obj_id < 0:
            raise ValueError(f"objects[{index}].obj_id must be a non-negative integer")
        if obj_id in seen_ids:
            raise ValueError(f"Duplicate obj_id: {obj_id}")
        seen_ids.add(obj_id)

        has_points = "points" in item or "labels" in item
        has_box = "box" in item
        if not has_points and not has_box:
            raise ValueError(f"obj_id={obj_id} must provide points+labels, box, or both")

        parsed_item: dict[str, Any] = {"obj_id": obj_id}

        if has_points:
            if "points" not in item or "labels" not in item:
                raise ValueError(f"obj_id={obj_id} must provide both points and labels")
            points = np.asarray(item.get("points"), dtype=np.float32)
            labels = np.asarray(item.get("labels"), dtype=np.int32)
            if points.ndim != 2 or points.shape[1:] != (2,) or len(points) == 0:
                raise ValueError(f"obj_id={obj_id} points must have shape (N, 2), N > 0")
            if labels.shape != (len(points),):
                raise ValueError(f"obj_id={obj_id} labels must have shape ({len(points)},)")
            if not np.isfinite(points).all():
                raise ValueError(f"obj_id={obj_id} points contain non-finite values")
            if not np.isin(labels, (0, 1)).all() or not np.any(labels == 1):
                raise ValueError(f"obj_id={obj_id} labels need at least one 1 and may only be 0/1")
            if not (
                np.all((0 <= points[:, 0]) & (points[:, 0] < width))
                and np.all((0 <= points[:, 1]) & (points[:, 1] < height))
            ):
                raise ValueError(f"obj_id={obj_id} contains a point outside the first frame")
            parsed_item.update({"points": points, "labels": labels})

        if has_box:
            box = np.asarray(item["box"], dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise ValueError(f"obj_id={obj_id} box must be four finite XYXY values")
            x0, y0, x1, y1 = box.tolist()
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError(f"obj_id={obj_id} box must lie inside the first frame")
            parsed_item["box"] = box

        parsed.append(parsed_item)

    return parsed


def prepare_output(output_dir: Path, obj_ids: list[int]) -> None:
    if output_dir.exists() and any(path.is_file() for path in output_dir.rglob("*")):
        raise ValueError(f"Output directory must contain no files: {output_dir}")
    for obj_id in obj_ids:
        (output_dir / "masks" / str(obj_id)).mkdir(parents=True, exist_ok=True)


def stage_for_sam2(frames: list[Path], directory: Path) -> None:
    """Expose any PNG/JPEG input as the numeric JPEG folder SAM 2 expects."""
    for frame_idx, source in enumerate(frames):
        os.symlink(source.resolve(), directory / f"{frame_idx:06d}.jpg")


def save_mask(mask_logits: Any, output_path: Path) -> None:
    mask = (mask_logits[0] > 0.0).detach().to("cpu").numpy().astype(np.uint8) * 255
    Image.fromarray(mask).save(output_path)


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.prompts.is_file():
        raise ValueError(f"Prompts JSON does not exist: {args.prompts}")

    frames, (height, width) = discover_frames(args.rgb_dir)
    prompts = load_prompts(args.prompts, height, width)
    obj_ids = [item["obj_id"] for item in prompts]
    prepare_output(args.output_dir, obj_ids)

    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")

    predictor = build_sam2_video_predictor(
        args.model_config,
        str(args.checkpoint),
        device=str(device),
    )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    manifest_frames: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="demo2gripper-sam2-") as staging:
        staging_dir = Path(staging)
        stage_for_sam2(frames, staging_dir)
        with torch.inference_mode(), autocast:
            state = predictor.init_state(
                str(staging_dir),
                offload_video_to_cpu=args.offload_video_to_cpu,
                offload_state_to_cpu=args.offload_state_to_cpu,
                async_loading_frames=args.async_loading_frames,
            )
            for prompt in prompts:
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=prompt["obj_id"],
                    points=prompt.get("points"),
                    labels=prompt.get("labels"),
                    box=prompt.get("box"),
                )

            for frame_idx, output_obj_ids, mask_logits in predictor.propagate_in_video(state):
                if set(output_obj_ids) != set(obj_ids):
                    raise RuntimeError(
                        f"SAM 2 returned object ids {list(output_obj_ids)}, expected {obj_ids}"
                    )
                source = frames[frame_idx]
                mask_paths: dict[str, str] = {}
                for obj_id, logits in zip(output_obj_ids, mask_logits, strict=True):
                    relative = Path("masks") / str(obj_id) / f"{source.stem}.png"
                    save_mask(logits, args.output_dir / relative)
                    mask_paths[str(obj_id)] = relative.as_posix()
                manifest_frames.append(
                    {
                        "frame_idx": frame_idx,
                        "source": source.name,
                        "masks": mask_paths,
                    }
                )

    if len(manifest_frames) != len(frames):
        raise RuntimeError(
            f"SAM 2 produced {len(manifest_frames)} frames for {len(frames)} inputs"
        )
    manifest = {
        "backend": "sam2",
        "display_name": f"SAM 2.1 {Path(args.model_config).stem}",
        "model_config": args.model_config,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "mask_encoding": {"dtype": "uint8", "background": 0, "foreground": 255},
        "image_size_hw": [height, width],
        "objects": [
            {
                "obj_id": item["obj_id"],
                "prompt_type": (
                    "box+points"
                    if "box" in item and "points" in item
                    else "box" if "box" in item else "points"
                ),
            }
            for item in prompts
        ],
        "frames": manifest_frames,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(frames)} frames for {len(obj_ids)} objects to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
