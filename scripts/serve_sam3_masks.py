#!/usr/bin/env python3
"""Serve a resident official SAM 3 image model for runtime mask bootstrap."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo2gripper.sam3_mask_service import (  # noqa: E402
    Sam3MaskInferenceService,
    create_mask_http_server,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--auth-token-env", default="D2G_MASK_TOKEN")
    return parser.parse_args()


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


class OfficialSam3Backend:
    def __init__(
        self,
        *,
        model_root: Path,
        checkpoint: Path,
        device: str,
        confidence_threshold: float,
    ) -> None:
        root = model_root.expanduser().resolve()
        weights = checkpoint.expanduser().resolve()
        if not (root / "sam3").is_dir():
            raise ValueError(f"model root does not contain sam3: {root}")
        if not weights.is_file():
            raise ValueError(f"SAM 3 checkpoint does not exist: {weights}")
        sys.path.insert(0, str(root))
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self._torch = torch
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        model = build_sam3_image_model(
            checkpoint_path=str(weights), device=str(self._device), load_from_HF=False
        )
        model = model.to(self._device)
        model.eval()
        self._processor = Sam3Processor(
            model,
            device=str(self._device),
            confidence_threshold=float(confidence_threshold),
        )
        self.metadata = {
            "name": "sam3",
            "implementation": "official_image_api",
            "checkpoint": str(weights),
            "device": str(self._device),
            "text_prompted": True,
        }

    def segment(self, rgb: np.ndarray, prompt: str) -> tuple[np.ndarray, float, np.ndarray]:
        torch = self._torch
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            state = self._processor.set_image(Image.fromarray(rgb, mode="RGB"))
            output = self._processor.set_text_prompt(state=state, prompt=prompt)
        masks = _to_numpy(output.get("masks"))
        boxes = _to_numpy(output.get("boxes")).reshape(-1, 4)
        scores = _to_numpy(output.get("scores")).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3 or not len(scores):
            raise RuntimeError(f"SAM 3 returned no mask for prompt {prompt!r}")
        if len(masks) != len(scores) or len(boxes) != len(scores):
            raise RuntimeError("SAM 3 returned inconsistent instances")
        index = int(np.argmax(scores))
        return masks[index] > 0.5, float(scores[index]), boxes[index]


def main() -> int:
    args = parse_args()
    backend = OfficialSam3Backend(
        model_root=args.model_root,
        checkpoint=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    token = os.environ.get(args.auth_token_env) or None
    server = create_mask_http_server(
        args.host,
        args.port,
        Sam3MaskInferenceService(backend),
        auth_token=token,
    )
    print(f"SAM 3 mask service listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
