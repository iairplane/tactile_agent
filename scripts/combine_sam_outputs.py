#!/usr/bin/env python3
"""Combine a SAM 2 hand track and SAM 2 or SAM 3 object masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-sam2-dir", type=Path, required=True)
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument(
        "--object-sam3-dir",
        type=Path,
        required=True,
        help="Object-mask directory produced by SAM 2 or SAM 3.",
    )
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    sam2 = load_json(args.hand_sam2_dir / "manifest.json")
    object_manifest = load_json(args.object_sam3_dir / "manifest.json")
    hand_frames = {str(record["source"]): record for record in sam2["frames"]}
    object_frames = {
        str(record["source"]): record for record in object_manifest["frames"]
    }
    if set(hand_frames) != set(object_frames):
        raise ValueError("Hand and object mask frame names differ")

    hand_output = args.output_dir / "masks" / "1"
    object_output = args.output_dir / "masks" / "2"
    hand_output.mkdir(parents=True, exist_ok=True)
    object_output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source in sorted(hand_frames, key=lambda name: int(Path(name).stem)):
        hand_relative = hand_frames[source]["masks"].get(str(args.hand_id))
        object_record = object_frames[source]
        object_relative = object_record.get("mask")
        if object_relative is None:
            object_relative = object_record.get("masks", {}).get(str(args.object_id))
        if hand_relative is None:
            raise ValueError(f"SAM 2 has no object {args.hand_id} for {source}")
        if object_relative is None:
            raise ValueError(
                f"Object manifest has no mask for object {args.object_id} in {source}"
            )
        hand_target = hand_output / source
        object_target = object_output / source
        shutil.copy2(args.hand_sam2_dir / hand_relative, hand_target)
        shutil.copy2(args.object_sam3_dir / object_relative, object_target)
        records.append(
            {
                "frame_idx": len(records),
                "source": source,
                "masks": {
                    "1": hand_target.relative_to(args.output_dir).as_posix(),
                    "2": object_target.relative_to(args.output_dir).as_posix(),
                },
            }
        )

    hand_source = str(sam2.get("display_name", sam2.get("backend", "SAM 2")))
    object_source = str(
        object_manifest.get("display_name", object_manifest.get("backend", "unknown"))
    )
    manifest = {
        "backend": f"sam2-hand-plus-{object_manifest.get('backend', 'object')}-object",
        "display_name": f"{hand_source} hand + {object_source} power drill",
        "sources": {
            "hand": {
                "directory": str(args.hand_sam2_dir),
                "backend": sam2.get("backend", "sam2"),
                "object_id": args.hand_id,
            },
            "object": {
                "directory": str(args.object_sam3_dir),
                "backend": object_manifest.get("backend", "unknown"),
                "object_id": args.object_id,
            },
        },
        "objects": [
            {"obj_id": 1, "label": "active hand", "source": hand_source},
            {"obj_id": 2, "label": "power drill", "source": object_source},
        ],
        "frames": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Combined {len(records)} frames in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
