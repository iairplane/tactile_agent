#!/usr/bin/env python3
"""Segment one text-prompted object per RGB frame with the official SAM 3 API."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


RGB_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help="Official SAM 3 checkout containing the sam3 Python package",
    )
    parser.add_argument("--text-prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Frame filename stem must be an integer: {path.name}") from exc


def discover_frames(rgb_dir: Path) -> tuple[list[Path], tuple[int, int]]:
    if not rgb_dir.is_dir():
        raise ValueError(f"RGB directory does not exist: {rgb_dir}")

    frames = sorted(
        (path for path in rgb_dir.iterdir() if path.suffix.lower() in RGB_SUFFIXES),
        key=_numeric_stem,
    )
    if not frames:
        raise ValueError(f"No PNG/JPEG frames found in {rgb_dir}")

    frame_numbers = [_numeric_stem(path) for path in frames]
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("RGB frame numbers are not unique")

    image_size: tuple[int, int] | None = None
    for path in frames:
        with Image.open(path) as image:
            image.load()
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


def prepare_output(output_dir: Path) -> Path:
    if output_dir.exists() and any(path.is_file() for path in output_dir.rglob("*")):
        raise ValueError(f"Output directory must contain no files: {output_dir}")
    mask_dir = output_dir / "masks" / "1"
    mask_dir.mkdir(parents=True, exist_ok=True)
    return mask_dir


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def select_best_instance(
    output: dict[str, Any], image_size: tuple[int, int]
) -> tuple[np.ndarray, list[float], float]:
    masks_value = output.get("masks")
    boxes_value = output.get("boxes")
    scores_value = output.get("scores")
    if masks_value is None or boxes_value is None or scores_value is None:
        raise RuntimeError("SAM 3 output is missing masks, boxes, or scores")

    masks = _to_numpy(masks_value)
    boxes = _to_numpy(boxes_value).reshape(-1, 4)
    scores = _to_numpy(scores_value).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise RuntimeError(f"SAM 3 masks must have shape (N,H,W), got {masks.shape}")
    if len(scores) == 0:
        raise RuntimeError("SAM 3 returned no instance for the text prompt")
    if len(masks) != len(scores) or len(boxes) != len(scores):
        raise RuntimeError("SAM 3 returned inconsistent mask, box, and score counts")

    best_index = int(np.argmax(scores))
    mask = masks[best_index] > 0.5
    if mask.shape != image_size:
        raise RuntimeError(
            f"SAM 3 mask shape {mask.shape} does not match RGB shape {image_size}"
        )
    if not np.any(mask):
        raise RuntimeError("SAM 3 highest-score instance has an empty mask")
    return mask, boxes[best_index].astype(float).tolist(), float(scores[best_index])


def save_overlay(
    image: Image.Image,
    mask: np.ndarray,
    box_xyxy: list[float],
    text_prompt: str,
    score: float,
    output_path: Path,
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    color = np.array([0, 210, 255], dtype=np.float32)
    rgb[mask] = np.rint(0.55 * rgb[mask] + 0.45 * color).astype(np.uint8)
    overlay = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(tuple(box_xyxy), outline=(255, 220, 0), width=3)
    draw.text(
        (box_xyxy[0] + 4, max(0.0, box_xyxy[1] - 14)),
        f"{text_prompt}  score={score:.3f}",
        fill=(255, 220, 0),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    overlay.save(output_path)


def main() -> int:
    args = parse_args()
    if not args.text_prompt.strip():
        raise ValueError("--text-prompt must not be empty")
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")
    model_root = args.model_root.resolve()
    if not (model_root / "sam3").is_dir():
        raise ValueError(f"Model root does not contain the sam3 package: {model_root}")

    frames, image_size = discover_frames(args.rgb_dir)
    mask_dir = prepare_output(args.output_dir)

    sys.path.insert(0, str(model_root))
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint.resolve()),
        device=str(device),
        load_from_HF=False,
    )
    model = model.to(device)
    model.eval()
    processor = Sam3Processor(model, device=str(device), confidence_threshold=0.0)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    frame_records: list[dict[str, Any]] = []
    with torch.inference_mode(), autocast:
        for frame_index, source in enumerate(frames):
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                started = time.perf_counter()
                state = processor.set_image(image)
                output = processor.set_text_prompt(
                    state=state, prompt=args.text_prompt.strip()
                )
                elapsed_s = time.perf_counter() - started
                mask, box_xyxy, score = select_best_instance(output, image_size)

                mask_relative = Path("masks") / "1" / f"{source.stem}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    args.output_dir / mask_relative
                )
                if frame_index == 0:
                    save_overlay(
                        image,
                        mask,
                        box_xyxy,
                        args.text_prompt.strip(),
                        score,
                        args.output_dir / "first_frame_overlay.png",
                    )

            frame_records.append(
                {
                    "frame_idx": frame_index,
                    "source": source.name,
                    "mask": mask_relative.as_posix(),
                    "score": score,
                    "box_xyxy": box_xyxy,
                    "elapsed_s": elapsed_s,
                }
            )
            print(
                f"{frame_index + 1}/{len(frames)} {source.name} "
                f"score={score:.3f} {elapsed_s:.2f}s"
            )

    manifest = {
        "backend": "Meta SAM 3/3.1 official image API",
        "display_name": "SAM 3 text-prompted segmentation",
        "selection": "highest-score instance per frame",
        "text_prompt": args.text_prompt.strip(),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_root": str(model_root),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mask_encoding": {"dtype": "uint8", "background": 0, "foreground": 255},
        "image_size_hw": list(image_size),
        "objects": [{"obj_id": 1, "prompt_type": "text"}],
        "first_frame_overlay": "first_frame_overlay.png",
        "frames": frame_records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(frames)} masks to {mask_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
